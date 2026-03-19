#include <filesystem>
#include <format>
#include <iostream>
#include <string_view>
#include <string>
#include <vector>

#include "pulse/model/qwen3/tokenizer.hpp"

namespace {

struct DemoCase {
    std::string_view name;
    std::string text;
    std::vector<pulse::i32> expected_tokens;
};

std::string join_arguments(int argc, char** argv, int begin) {
    std::string text;
    for (int i = begin; i < argc; ++i) {
        if (!text.empty()) {
            text.push_back(' ');
        }
        text += argv[i];
    }
    return text;
}

void print_tokens(const std::vector<pulse::i32>& tokens) {
    std::cout << "[";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << tokens[i];
    }
    std::cout << "]";
}

void print_token_pieces(const pulse::model::Qwen3Tokenizer& tokenizer,
                        const std::vector<pulse::i32>& tokens) {
    std::cout << "[";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::format("{}:\"{}\"", tokens[i], tokenizer.decode_token(tokens[i]));
    }
    std::cout << "]";
}

bool run_case(const pulse::model::Qwen3Tokenizer& tokenizer,
              const DemoCase& demo_case) {
    const auto& text = demo_case.text;
    const auto tokens = tokenizer.encode(text);
    const auto decoded = tokenizer.decode(tokens);
    const bool matched = decoded == text;
    const bool has_expected_tokens = !demo_case.expected_tokens.empty();
    const bool token_match = !has_expected_tokens || tokens == demo_case.expected_tokens;
    const bool passed = matched && token_match;

    std::cout << std::format("[{}]\n", demo_case.name);
    std::cout << std::format("input  : {}\n", text);
    std::cout << "tokens : ";
    print_tokens(tokens);
    std::cout << '\n';
    std::cout << "pieces : ";
    print_token_pieces(tokenizer, tokens);
    std::cout << '\n';
    std::cout << std::format("count  : {}\n", tokens.size());
    std::cout << std::format("decode : {}\n", decoded);
    std::cout << std::format("match  : {}\n", matched ? "true" : "false");
    if (has_expected_tokens) {
        std::cout << "expect : ";
        print_tokens(demo_case.expected_tokens);
        std::cout << '\n';
        std::cout << std::format("token_check : {}\n", token_match ? "true" : "false");
    } else {
        std::cout << "token_check : skipped\n";
    }
    std::cout << std::format("status : {}\n\n", passed ? "pass" : "fail");

    return passed;
}

bool run_special_token_check(const pulse::model::Qwen3Tokenizer& tokenizer,
                             std::string_view token,
                             pulse::i32 expected_id) {
    const auto encoded = tokenizer.encode(std::string(token));
    const auto actual_id = tokenizer.token_id(token);
    const bool id_match = actual_id == expected_id;
    const bool single_token = encoded.size() == 1 && encoded.front() == expected_id;
    const bool passed = id_match && single_token;

    std::cout << std::format("[special:{}]\n", token);
    std::cout << std::format("token_id        : {}\n", actual_id);
    std::cout << std::format("expected_id     : {}\n", expected_id);
    std::cout << std::format("id_check        : {}\n", id_match ? "true" : "false");
    std::cout << std::format("single_token_ok : {}\n", single_token ? "true" : "false");
    std::cout << "encoded         : ";
    print_tokens(encoded);
    std::cout << std::format("\nstatus          : {}\n\n", passed ? "pass" : "fail");

    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    const fs::path model_path = argc > 1 ? fs::path(argv[1]) : fs::path("../../model");

    auto tokenizer_result = pulse::model::Qwen3Tokenizer::load(model_path);
    if (!tokenizer_result) {
        std::cerr << std::format("failed to load tokenizer: {}\n",
                                 tokenizer_result.error().message());
        return 1;
    }

    auto tokenizer = std::move(tokenizer_result.value());

    std::cout << std::format("model path : {}\n", model_path.string());
    std::cout << std::format("vocab size : {}\n", tokenizer.vocab_size());
    std::cout << std::format("eos id     : {}\n", tokenizer.eos_id());
    std::cout << std::format("pad id     : {}\n", tokenizer.pad_id());
    std::cout << std::format("im_start id: {}\n", tokenizer.im_start_id());
    std::cout << std::format("im_end id  : {}\n\n", tokenizer.im_end_id());

    if (argc > 2) {
        const DemoCase custom_case{
            .name = "custom",
            .text = join_arguments(argc, argv, 2),
            .expected_tokens = {},
        };
        return run_case(tokenizer, custom_case) ? 0 : 1;
    }

    const std::vector<DemoCase> demo_cases = {
        {
            "basic_english",
            "Hello, world!",
            {9707, 11, 1879, 0},
        },
        {
            "english_contractions",
            "I'm testing Qwen3 tokenizer and we'll verify it's correct.",
            {40, 2776, 7497, 1207, 16948, 18, 45958, 323, 582, 6, 654, 10146, 432, 6, 82, 4396, 13},
        },
        {
            "whitespace_and_tabs",
            "  leading spaces\tand tabs\nwith blank lines\n\nend",
            {256, 20654, 12621, 52477, 22398, 198, 4197, 10113, 5128, 271, 408},
        },
        {
            "basic_chinese",
            "你好，世界！",
            {108386, 3837, 99489, 6313},
        },
        {
            "mixed_language",
            "Qwen3 可以处理 English、中文、12345 和 emoji 😀🚀。",
            {48, 16948, 18, 26853, 107, 23031, 54542, 6364, 5373, 104811, 5373, 16, 17, 18, 19, 20, 58143, 42365, 90316, 145836, 1773},
        },
        {
            "punctuation_dense",
            "Wait... what?! -- yes; {json: \"like\", list: [1,2,3]}",
            {14190, 13, 13, 13, 1128, 30, 0, 1177, 9834, 26, 314, 2236, 25, 330, 4803, 1, 11, 1140, 25, 508, 16, 11, 17, 11, 18, 60, 92},
        },
        {
            "chat_template_fragment",
            "<|im_start|>user\n请用一句话介绍 Qwen3 tokenizer。\n<|im_end|>\n<|im_start|>assistant\n",
            {151644, 872, 198, 14880, 11622, 105321, 100157, 1207, 16948, 18, 45958, 1773, 198, 151645, 198, 151644, 77091, 198},
        },
        {
            "thinking_fragment",
            "<think>\nstep by step\n</think>\nfinal answer",
            {151667, 198, 9520, 553, 3019, 198, 151668, 198, 11822, 4226},
        },
        {
            "tool_call_fragment",
            "<tool_call>\n{\"name\":\"sum\",\"arguments\":{\"a\":1,\"b\":2}}\n</tool_call>",
            {151657, 198, 90, 1, 606, 1, 25, 1, 1242, 1, 11, 1, 16370, 1, 25, 90, 1, 64, 1, 25, 16, 11, 1, 65, 1, 25, 17, 92, 92, 198, 151658},
        },
        {
            "tool_response_fragment",
            "<tool_response>\n{\"result\":3}\n</tool_response>",
            {151665, 198, 90, 1, 1382, 1, 25, 18, 92, 198, 151666},
        },
        {
            "adjacent_special_tokens",
            "<|im_start|><think></think><|im_end|>",
            {151644, 151667, 151668, 151645},
        },
        {
            "fim_tokens",
            "<|fim_prefix|>int add(int a, int b) {<|fim_suffix|>return a + b;<|fim_middle|>\n",
            {151659, 396, 912, 7, 396, 264, 11, 526, 293, 8, 314, 151661, 689, 264, 488, 293, 26, 151660, 198},
        },
    };

    bool all_passed = true;
    for (const auto& demo_case : demo_cases) {
        all_passed = run_case(tokenizer, demo_case) && all_passed;
    }

    all_passed = run_special_token_check(tokenizer, "<|im_start|>", tokenizer.im_start_id()) && all_passed;
    all_passed = run_special_token_check(tokenizer, "<|im_end|>", tokenizer.im_end_id()) && all_passed;
    all_passed = run_special_token_check(tokenizer, "<think>", tokenizer.think_start_id()) && all_passed;
    all_passed = run_special_token_check(tokenizer, "</think>", tokenizer.think_end_id()) && all_passed;

    std::cout << std::format("[summary]\nresult : {}\n", all_passed ? "pass" : "fail");
    return all_passed ? 0 : 1;
}
