#include "pulse/model/qwen3/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace pulse::model {

namespace {

using json = nlohmann::json;


[[nodiscard]] Result<std::string> read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return Err<std::string>(ErrorCode::OpenFileError, std::format("Failed to open '{}'", path.string()));
    }

    file.seekg(0, std::ios::end);
    const auto file_size_raw = file.tellg();
    if (file_size_raw < 0) {
        return Err<std::string>(ErrorCode::OpenFileError,
                                std::format("Failed to get size of '{}'", path.string()));
    }

    const auto file_size = static_cast<usize>(file_size_raw);
    file.seekg(0, std::ios::beg);

    std::string content(file_size, '\0');
    if (file_size > 0) {
        file.read(content.data(), static_cast<std::streamsize>(file_size));
    }
    return Ok(std::move(content));
}

[[nodiscard]] std::filesystem::path resolve_model_dir(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path;
    }

    return path.parent_path();
}

[[nodiscard]] usize utf8_char_len(u8 byte) noexcept {
    if (byte < 0x80u) {
        return 1;
    }

    if (byte < 0xC0u) {
        return 1;
    }

    if (byte < 0xE0u) {
        return 2;
    }

    if (byte < 0xF0u) {
        return 3;
    }

    return 4;
}

[[nodiscard]] std::string codepoint_to_utf8(i32 codepoint) {
    std::string text;
    if (codepoint < 0x80) {
        text.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        text.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint < 0x10000) {
        text.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        text.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }

    return text;
}

[[nodiscard]] std::vector<std::string> utf8_chars(std::string_view text) {
    std::vector<std::string> chars;
    chars.reserve(text.size());

    usize offset = 0;
    while (offset < text.size()) {
        usize len = utf8_char_len(static_cast<u8>(text[offset]));
        if (offset + len > text.size()) {
            len = 1;
        }
        chars.emplace_back(text.substr(offset, len));
        offset += len;
    }

    return chars;
}

[[nodiscard]] bool is_ascii_letter(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

}  // namespace

Result<Qwen3Tokenizer> Qwen3Tokenizer::load(const std::filesystem::path& path) {
    const auto model_dir = resolve_model_dir(path);
    const auto vocab_path = model_dir / "vocab.json";
    const auto merges_path = model_dir / "merges.txt";

    if (!std::filesystem::exists(vocab_path)) {
        return Err<Qwen3Tokenizer>(
            ErrorCode::OpenFileError,
            std::format("Tokenizer vocab file '{}' does not exist", vocab_path.string()));
    }

    if (!std::filesystem::exists(merges_path)) {
        return Err<Qwen3Tokenizer>(
            ErrorCode::OpenFileError,
            std::format("Tokenizer merges file '{}' does not exist", merges_path.string()));
    }

    Qwen3Tokenizer tokenizer;

    // 顺序不能乱：
    // 1. 先建 byte mapping
    // 2. 再加载 vocab / merges
    // 3. 最后补 special token 元信息
    //
    // 因为 encode/decode 的普通路径依赖前两步，special token 则依赖词表最终状态。
    tokenizer.init_byte_mapping();

    auto vocab_result = tokenizer.load_vocab(vocab_path);
    if (!vocab_result) {
        return Err<Qwen3Tokenizer>(std::move(vocab_result.error()));
    }

    auto merges_result = tokenizer.load_merges(merges_path);
    if (!merges_result) {
        return Err<Qwen3Tokenizer>(std::move(merges_result.error()));
    }

    struct SpecialTokenDef {
        const char* content;
        i32 fallback_id;
    };

    // Qwen3-0.6B 当前模型目录中的 special token 定义。
    // 这里沿用参考实现的策略: 优先从 vocab.json 中查实际 id，找不到再退回到已知固定 id。
    constexpr SpecialTokenDef kSpecialTokens[] = {
        {"<|endoftext|>",        151643},
        {"<|im_start|>",         151644},
        {"<|im_end|>",           151645},
        {"<|object_ref_start|>", 151646},
        {"<|object_ref_end|>",   151647},
        {"<|box_start|>",        151648},
        {"<|box_end|>",          151649},
        {"<|quad_start|>",       151650},
        {"<|quad_end|>",         151651},
        {"<|vision_start|>",     151652},
        {"<|vision_end|>",       151653},
        {"<|vision_pad|>",       151654},
        {"<|image_pad|>",        151655},
        {"<|video_pad|>",        151656},
        {"<tool_call>",          151657},
        {"</tool_call>",         151658},
        {"<|fim_prefix|>",       151659},
        {"<|fim_middle|>",       151660},
        {"<|fim_suffix|>",       151661},
        {"<|fim_pad|>",          151662},
        {"<|repo_name|>",        151663},
        {"<|file_sep|>",         151664},
        {"<tool_response>",      151665},
        {"</tool_response>",     151666},
        {"<think>",              151667},
        {"</think>",             151668},
    };

    for (const auto& special : kSpecialTokens) {
        // 优先尊重 vocab.json 中真实存在的 id，fallback 只是防御性兜底。
        i32 token_id = special.fallback_id;
        if (auto it = tokenizer.piece_to_id_.find(special.content); it != tokenizer.piece_to_id_.end()) {
            token_id = it->second;
        }

        tokenizer.special_tokens_[special.content] = token_id;
        tokenizer.special_token_list_.emplace_back(special.content, token_id);
        tokenizer.piece_to_id_[special.content] = token_id;

        const auto size = static_cast<usize>(token_id) + 1;
        if (tokenizer.id_to_piece_.size() < size) {
            tokenizer.id_to_piece_.resize(size);
        }
        tokenizer.id_to_piece_[static_cast<usize>(token_id)] = special.content;
    }

    std::ranges::sort(tokenizer.special_token_list_, [](const auto& lhs, const auto& rhs) {
        if (lhs.first.size() != rhs.first.size()) {
            return lhs.first.size() > rhs.first.size();
        }
        return lhs.first < rhs.first;
    });

    tokenizer.pad_id_ = tokenizer.special_tokens_.at("<|endoftext|>");
    tokenizer.im_start_id_ = tokenizer.special_tokens_.at("<|im_start|>");
    tokenizer.im_end_id_ = tokenizer.special_tokens_.at("<|im_end|>");
    // Qwen3 推理里通常把 <|im_end|> 当作 EOS 使用。
    tokenizer.eos_id_ = tokenizer.im_end_id_;
    tokenizer.think_start_id_ = tokenizer.special_tokens_.at("<think>");
    tokenizer.think_end_id_ = tokenizer.special_tokens_.at("</think>");
    tokenizer.loaded_ = true;

    return Ok(std::move(tokenizer));
}

std::vector<i32> Qwen3Tokenizer::encode(std::string_view text) const {
    if (!loaded_ || text.empty()) {
        return {};
    }

    std::vector<i32> token_ids;
    token_ids.reserve(text.size() / 2);

    usize offset = 0;
    while (offset < text.size()) {
        const auto [special_length, special_id] = match_special_token(text, offset);
        if (special_length > 0) {
            // special token 一旦匹配，直接落成单个 id，不再参与预分词和 BPE。
            token_ids.push_back(special_id);
            offset += special_length;
            continue;
        }

        usize next_special = text.size();
        for (const auto& [special_token, special_id_unused] : special_token_list_) {
            static_cast<void>(special_id_unused);
            const auto found = text.find(special_token, offset);
            if (found != std::string_view::npos && found < next_special) {
                next_special = found;
            }
        }

        // 这一步把“下一个 special token 之前的普通文本”单独拿出来处理，
        // 这样可以同时满足：
        // 1. special token 原子匹配
        // 2. 普通文本仍然走完整 byte-level BPE
        const auto chunk = text.substr(offset, next_special - offset);
        const auto pieces = pretokenize(chunk);
        for (const auto& piece : pieces) {
            const auto ids = bpe(byte_encode(piece));
            token_ids.insert(token_ids.end(), ids.begin(), ids.end());
        }

        offset = next_special;
    }

    return token_ids;
}

std::string Qwen3Tokenizer::decode(const std::vector<i32>& tokens) const {
    // decode(vector) 只是顺序拼接每个 token 的解码结果。
    // 这里不做任何“后处理”或 trim，尽量保持与词表的一一映射关系。
    std::string text;
    text.reserve(tokens.size() * 4);

    for (i32 token_id : tokens) {
        text += decode_token(token_id);
    }

    return text;
}

std::string Qwen3Tokenizer::decode_token(i32 token_id) const {
    if (!loaded_ || token_id < 0 || static_cast<usize>(token_id) >= id_to_piece_.size()) {
        return {};
    }

    const auto& piece = id_to_piece_[static_cast<usize>(token_id)];
    if (special_tokens_.contains(piece)) {
        // special token 在词表里就是协议字符串本身，
        // 如果继续走 byte_decode 反而会破坏它的可读性。
        return piece;
    }

    return byte_decode(piece);
}

i32 Qwen3Tokenizer::token_id(std::string_view token) const {
    if (auto it = piece_to_id_.find(std::string(token)); it != piece_to_id_.end()) {
        return it->second;
    }
    return -1;
}

Result<void> Qwen3Tokenizer::load_vocab(const std::filesystem::path& path) {
    auto file_result = read_text_file(path);
    if (!file_result) {
        return Err<void>(std::move(file_result.error()));
    }

    // vocab.json 体积不算小，但这里直接用 nlohmann_json 可读性更好，
    // 也符合当前项目已经引入该依赖的事实，没有必要为了这一步手写 JSON 解析。
    auto vocab_json = json::parse(file_result.value(), nullptr, false, true);
    if (vocab_json.is_discarded() || !vocab_json.is_object()) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("Failed to parse vocab json '{}'", path.string()));
    }

    // 先记录最大 id，再一次性建立 id -> piece 反向表。
    i32 max_id = -1;
    for (const auto& [piece, id_value] : vocab_json.items()) {
        if (!id_value.is_number_integer()) {
            return Err<void>(ErrorCode::InvalidArgument,
                             std::format("Invalid vocab id for piece '{}' in '{}'", piece, path.string()));
        }

        const auto token_id = id_value.get<i32>();
        piece_to_id_[piece] = token_id;
        if (token_id > max_id) {
            max_id = token_id;
        }
    }

    if (piece_to_id_.empty()) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("No vocab entries found in '{}'", path.string()));
    }

    id_to_piece_.assign(static_cast<usize>(max_id) + 1, std::string{});
    for (const auto& [piece, token_id] : piece_to_id_) {
        if (token_id >= 0 && static_cast<usize>(token_id) < id_to_piece_.size()) {
            id_to_piece_[static_cast<usize>(token_id)] = piece;
        }
    }

    return Ok();
}

Result<void> Qwen3Tokenizer::load_merges(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return Err<void>(ErrorCode::OpenFileError, std::format("Failed to open '{}'", path.string()));
    }

    // merges.txt 中每一行的行号就是 rank。
    // rank 越小，BPE 合并时优先级越高。
    std::string line;
    i32 rank = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (line.find(' ') == std::string::npos) {
            continue;
        }

        merges_[line] = rank++;
    }

    if (merges_.empty()) {
        return Err<void>(ErrorCode::InvalidArgument,
                         std::format("No merge entries found in '{}'", path.string()));
    }

    return Ok();
}

void Qwen3Tokenizer::init_byte_mapping() {
    // GPT2 byte_encoder: 将每个 byte (0-255) 映射为一个 Unicode codepoint
    //
    // 可打印字节 (33-126, 161-172, 174-255) 直接映射为同值 codepoint
    // 其余字节 (0-32, 127-160, 173) 依次映射为 256, 257, ...
    //
    // 映射后的 codepoint 都是可打印的 Unicode 字符
    std::vector<i32> bytes;
    std::vector<i32> codepoints;

    bytes.reserve(256);
    codepoints.reserve(256);

    auto push_range = [&](i32 begin, i32 end) {
        for (i32 value = begin; value <= end; ++value) {
            bytes.push_back(value);
            codepoints.push_back(value);
        }
    };

    push_range(33, 126);
    push_range(161, 172);
    push_range(174, 255);

    i32 next_codepoint_offset = 0;

    for (i32 byte = 0; byte < 256; ++byte) {
        if (std::ranges::find(bytes, byte) != bytes.end()) {
            continue;
        }

        bytes.push_back(byte);
        codepoints.push_back(256 + next_codepoint_offset);
        ++next_codepoint_offset;
    }

    for (usize i = 0; i < bytes.size(); ++i) {
        const auto utf8 = codepoint_to_utf8(codepoints[i]);
        byte_to_unicode_[static_cast<usize>(bytes[i])] = utf8;
        unicode_to_byte_[utf8] = static_cast<u8>(bytes[i]);
    }
}

std::string Qwen3Tokenizer::byte_encode(std::string_view text) const {
    std::string encoded;
    encoded.reserve(text.size() * 2);

    for (u8 byte : text) {
        encoded += byte_to_unicode_[byte];
    }

    return encoded;
}

std::string Qwen3Tokenizer::byte_decode(std::string_view text) const {
    std::string decoded;
    decoded.reserve(text.size());

    usize offset = 0;
    while (offset < text.size()) {
        usize len = utf8_char_len(static_cast<u8>(text[offset]));
        if (offset + len > text.size()) {
            break;
        }

        const std::string piece{text.substr(offset, len)};
        if (auto it = unicode_to_byte_.find(piece); it != unicode_to_byte_.end()) {
            decoded.push_back(static_cast<char>(it->second));
        }
        offset += len;
    }

    return decoded;
}

std::vector<std::string> Qwen3Tokenizer::pretokenize(std::string_view text) const {
    std::vector<std::string> pieces;
    usize offset = 0;

    while (offset < text.size()) {
        const auto byte = static_cast<u8>(text[offset]);

        if (byte == '\n' || byte == '\r') {
            std::string piece;
            while (offset < text.size() && (text[offset] == '\n' || text[offset] == '\r')) {
                piece.push_back(text[offset]);
                ++offset;
            }
            pieces.push_back(std::move(piece));
            continue;
        }

        if (byte == ' ' || byte == '\t') {
            if (offset + 1 < text.size()) {
                const auto next = static_cast<u8>(text[offset + 1]);

                if (is_ascii_letter(static_cast<char>(next))) {
                    // " word"
                    std::string piece;
                    piece.push_back(text[offset++]);
                    while (offset < text.size() && is_ascii_letter(text[offset])) {
                        piece.push_back(text[offset++]);
                    }
                    pieces.push_back(std::move(piece));
                    continue;
                }

                if (next >= 0x80u) {
                    // "chinese" / " emoji"
                    std::string piece;
                    piece.push_back(text[offset++]);
                    while (offset < text.size() && static_cast<u8>(text[offset]) >= 0x80u) {
                        usize len = utf8_char_len(static_cast<u8>(text[offset]));
                        if (offset + len > text.size()) {
                            break;
                        }
                        piece.append(text.substr(offset, len));
                        offset += len;
                    }
                    pieces.push_back(std::move(piece));
                    continue;
                }

                if (std::isdigit(next) != 0) {
                    pieces.emplace_back(text.substr(offset, 1));
                    ++offset;
                    continue;
                }

                if (std::isspace(next) == 0) {
                    // " !" / "..." / "，"
                    std::string piece;
                    piece.push_back(text[offset++]);
                    while (offset < text.size() && std::isspace(static_cast<u8>(text[offset])) == 0 &&
                           !is_ascii_letter(text[offset]) &&
                           std::isdigit(static_cast<u8>(text[offset])) == 0 &&
                           static_cast<u8>(text[offset]) < 0x80u) {
                        piece.push_back(text[offset++]);
                    }
                    pieces.push_back(std::move(piece));
                    continue;
                }
            }

            std::string piece;
            while (offset < text.size() && (text[offset] == ' ' || text[offset] == '\t')) {
                piece.push_back(text[offset++]);
            }
            pieces.push_back(std::move(piece));
            continue;
        }

        if (std::isdigit(byte) != 0) {
            pieces.emplace_back(text.substr(offset, 1));
            ++offset;
            continue;
        }

        if (is_ascii_letter(static_cast<char>(byte))) {
            std::string piece;
            while (offset < text.size() && is_ascii_letter(text[offset])) {
                piece.push_back(text[offset++]);
            }

            if (offset < text.size() && text[offset] == '\'') {
                std::string suffix = "'";
                usize next = offset + 1;
                while (next < text.size() && is_ascii_letter(text[next]) && next - offset <= 3) {
                    suffix.push_back(text[next]);
                    ++next;
                }

                auto lower = suffix.substr(1);
                std::ranges::transform(lower, lower.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });

                if (lower == "s" || lower == "t" || lower == "re" || lower == "ve" || lower == "m" ||
                    lower == "ll" || lower == "d") {
                    pieces.push_back(std::move(piece));
                    pieces.push_back(std::move(suffix));
                    offset = next;
                    continue;
                }
            }

            pieces.push_back(std::move(piece));
            continue;
        }

        if (byte >= 0x80u) {
            std::string piece;
            while (offset < text.size() && static_cast<u8>(text[offset]) >= 0x80u) {
                usize len = utf8_char_len(static_cast<u8>(text[offset]));
                if (offset + len > text.size()) {
                    break;
                }
                piece.append(text.substr(offset, len));
                offset += len;
            }
            if (!piece.empty()) {
                pieces.push_back(std::move(piece));
            }
            continue;
        }

        pieces.emplace_back(text.substr(offset, 1));
        ++offset;
    }

    return pieces;
}

std::vector<i32> Qwen3Tokenizer::bpe(std::string_view piece) const {
    auto symbols = utf8_chars(piece);
    if (symbols.empty()) {
        return {};
    }

    if (symbols.size() == 1) {
        if (auto it = piece_to_id_.find(symbols.front()); it != piece_to_id_.end()) {
            return {it->second};
        }
        return {};
    }

    while (symbols.size() > 1) {
        i32 best_rank = INT_MAX;
        isize best_index = -1;

        for (usize i = 0; i + 1 < symbols.size(); ++i) {
            const auto key = symbols[i] + " " + symbols[i + 1];
            if (auto it = merges_.find(key); it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = static_cast<isize>(i);
            }
        }

        if (best_index < 0) {
            break;
        }

        const auto merged =
            symbols[static_cast<usize>(best_index)] + symbols[static_cast<usize>(best_index) + 1];
        symbols[static_cast<usize>(best_index)] = std::move(merged);
        symbols.erase(symbols.begin() + best_index + 1);
    }

    std::vector<i32> ids;
    ids.reserve(symbols.size());
    for (const auto& symbol : symbols) {
        if (auto it = piece_to_id_.find(symbol); it != piece_to_id_.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::pair<usize, i32> Qwen3Tokenizer::match_special_token(std::string_view text, usize offset) const {
    for (const auto& [token, token_id] : special_token_list_) {
        if (offset + token.size() > text.size()) {
            continue;
        }

        if (text.compare(offset, token.size(), token) == 0) {
            return {token.size(), token_id};
        }
    }

    return {0, -1};
}

}  // namespace pulse::model
