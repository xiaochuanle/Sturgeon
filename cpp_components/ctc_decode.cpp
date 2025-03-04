
#include <iostream>
#include <torch/torch.h>
#include <cstring>
#include <cmath>
#include <cassert>

extern "C" {
    uint8_t get_qual(float x) {
        x = x < 1e-7 ? 1e-7 : x;
        x = x > (1.0 - 1e-7) ? (1.0 - 1e-7) : x;
        return static_cast<uint8_t>(-10 * std::log10(1 - x) + 33);
    }

    bool ctc_veterbi_decode(float * inputs, uint8_t *seqs, uint8_t * moves, uint8_t * quals, int T, int N, int C) {

//        auto t1 = std::chrono::high_resolution_clock::now();

        torch::Tensor inputs_t = torch::from_blob(inputs, {T, N, C}, torch::kFloat32);

//        auto t2 = std::chrono::high_resolution_clock::now();
        torch::Tensor soft_inputs = torch::exp(inputs_t).permute({1, 0, 2}).contiguous(); // T x N x C -> N x T x C
        torch::Tensor logits = soft_inputs.argmax(2).contiguous(); // N x T x C -> N x T

//        auto t3 = std::chrono::high_resolution_clock::now();

        float * soft_inputs_ptr = soft_inputs.data_ptr<float>();
        long * logits_ptr = logits.data_ptr<long>();

        // todo: make this outer loop parallel in the future
        for (int i = 0; i < N; i++) {
            if (logits_ptr[i * T] != 0) {
                seqs[i * T + 0] = logits_ptr[i * T];
                moves[i * T + 0] = 1;
                int64_t k = logits_ptr[i * T + 0];
                quals[i * T + 0] = get_qual(soft_inputs_ptr[i * T * C + k]);
//                quals[i * T + 0] = get_qual(soft_inputs_ptr[i * T * C + j * C + logits_ptr[i * T + 0]]);
            }
            for (int j = 1; j < T; j ++) {
                if (logits_ptr[i * T + j] != 0 && \
                        logits_ptr[i * T + j - 1] != logits_ptr[i * T + j]) {
                    seqs[i * T + j] = logits_ptr[i * T + j];
                    moves[i * T + j] = 1;
                    int64_t k = logits_ptr[i * T + j];
                    quals[i * T + j] = get_qual(soft_inputs_ptr[i * T * C + j * C + k]);
//                    quals[i * T + j] = get_qual(soft_inputs_ptr[i * T * C + j * C + logits_ptr[i * T + j]]);
                }
            }
        }

//        auto t4 = std::chrono::high_resolution_clock::now();
//
//
//        auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
//        auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
//        auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
//        std::cout << "time: " << d1.count() << "ms, " << d2.count() << "ms, " << d3.count() << "ms" << std::endl;
        return true;
    }
}