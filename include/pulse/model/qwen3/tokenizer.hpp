#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulse/core/error.hpp"
#include "pulse/core/types.hpp"

namespace pulse::model {

class Qwen3Tokenizer {
public:
    Qwen3Tokenizer() = default;
    ~Qwen3Tokenizer() = default;

    Qwen3Tokenizer(const Qwen3Tokenizer&) = delete;
    Qwen3Tokenizer& operator=(const Qwen3Tokenizer&) = delete;

    Qwen3Tokenizer(Qwen3Tokenizer&&) noexcept = default;
    Qwen3Tokenizer& operator=(Qwen3Tokenizer&&) noexcept = default;


    [[nodiscard]] static Result<Qwen3Tokenizer> load(const std::filesystem::path& path);

    // 编码流程:
    //   text
    //   -> special token greedy match
    //   -> pretokenize
    //   -> byte_encode
    //   -> BPE merge
    //   -> ids
    //
    [[nodiscard]] std::vector<i32> encode(std::string_view text) const;

    // 解码流程:
    //   ids
    //   -> vocab piece
    //   -> special token 直接返回原文
    //   -> 普通 piece 走 byte_decode
    [[nodiscard]] std::string decode(const std::vector<i32>& tokens) const;
    [[nodiscard]] std::string decode_token(i32 token_id) const;

    [[nodiscard]] i32 token_id(std::string_view token) const;

    [[nodiscard]] i32 vocab_size() const noexcept {
        return static_cast<i32>(id_to_piece_.size());
    }

    [[nodiscard]] i32 eos_id() const noexcept {
        return eos_id_;
    }

    [[nodiscard]] i32 pad_id() const noexcept {
        return pad_id_;
    }

    [[nodiscard]] i32 im_start_id() const noexcept {
        return im_start_id_;
    }

    [[nodiscard]] i32 im_end_id() const noexcept {
        return im_end_id_;
    }

    [[nodiscard]] i32 think_start_id() const noexcept {
        return think_start_id_;
    }

    [[nodiscard]] i32 think_end_id() const noexcept {
        return think_end_id_;
    }

private:
    [[nodiscard]] Result<void> load_vocab(const std::filesystem::path& path);


    [[nodiscard]] Result<void> load_merges(const std::filesystem::path& path);


    void init_byte_mapping();


    [[nodiscard]] std::string byte_encode(std::string_view text) const;

    [[nodiscard]] std::string byte_decode(std::string_view text) const;


    [[nodiscard]] std::vector<std::string> pretokenize(std::string_view text) const;

    [[nodiscard]] std::vector<i32> bpe(std::string_view piece) const;


    [[nodiscard]] std::pair<usize, i32> match_special_token(std::string_view text, usize offset) const;

    bool loaded_ = false;


    std::unordered_map<std::string, i32> piece_to_id_;
    std::vector<std::string> id_to_piece_;

    std::unordered_map<std::string, i32> special_tokens_;
    std::vector<std::pair<std::string, i32>> special_token_list_;

    std::unordered_map<std::string, i32> merges_;

    std::array<std::string, 256> byte_to_unicode_;
    std::unordered_map<std::string, u8> unicode_to_byte_;

    i32 eos_id_ = -1;
    i32 pad_id_ = -1;
    i32 im_start_id_ = -1;
    i32 im_end_id_ = -1;
    i32 think_start_id_ = -1;
    i32 think_end_id_ = -1;
};

}  // namespace pulse::model
