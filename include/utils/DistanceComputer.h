#pragma once
#include <vector>
#include <immintrin.h>
#include <stdexcept> 

class DistanceComputer {
    public:
        virtual float symmetric_dis(int idx_a, int idx_b) = 0;
        virtual void set_query(const float* x) = 0;
        virtual void set_query_storage(int idx) = 0;
        virtual float operator()(int index) = 0;
        virtual ~DistanceComputer() = default;
};

class GenericDistanceComputerL2 : public DistanceComputer {
    int d;
    const std::vector<float>& storage;
    const float* q;

public:
    GenericDistanceComputerL2(const std::vector<float>& storage, int d) 
        : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = q[i] - storage[index * d + i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = storage[idx_a * d + i] - storage[idx_b * d + i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    void set_query(const float* x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q = &storage[idx * d];
    }
};


class GenericDistanceComputerIP : public DistanceComputer {
    int d;  // the vector dimension
    const std::vector<float>& storage;  // vector storage
    const float* q;  // the query vector

public:
    GenericDistanceComputerIP(const std::vector<float>& storage, int d) 
        : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        if (q == nullptr) {
            throw std::runtime_error("Query pointer is not set");
        }
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += q[i] * storage[index * d + i];
        }
        return -sum;
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += storage[idx_a * d + i] * storage[idx_b * d + i];
        }
        return -sum;
    }

    void set_query(const float* x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q = &storage[idx * d];
    }
};



// AVX horizental sum
inline float hsum256_ps(__m256 v) {
    __m128 v128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    v128 = _mm_add_ps(v128, _mm_movehdup_ps(v128));
    v128 = _mm_add_ss(v128, _mm_movehl_ps(v128, v128));
    return _mm_cvtss_f32(v128);
}

class GenericDistanceComputerIP_AVX2 : public DistanceComputer {
    
    int d;  // vector dimension
    const std::vector<float>& storage;  // vector datas
    const float* q;  // pointer for query vector

    public:
        GenericDistanceComputerIP_AVX2(const std::vector<float>& storage, int d) 
            : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        if (q == nullptr) throw std::runtime_error("Query pointer not set");
        
        const float* base = &storage[index * d];
        __m256 sum8 = _mm256_setzero_ps();
        int i = 0;

        // main loop: AVX parallel
        for (; i + 7 < d; i += 8) {
            __m256 q_vec = _mm256_loadu_ps(q + i);
            __m256 b_vec = _mm256_loadu_ps(base + i);
            sum8 = _mm256_fmadd_ps(q_vec, b_vec, sum8);
        }

        // residue elements
        float sum = hsum256_ps(sum8);
        for (; i < d; ++i) {
            sum += q[i] * base[i];
        }

        return -sum;  // the minus is for distance comparison
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        const float* a = &storage[idx_a * d];
        const float* b = &storage[idx_b * d];
        __m256 sum8 = _mm256_setzero_ps();
        int i = 0;

        for (; i + 7 < d; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 b_vec = _mm256_loadu_ps(b + i);
            sum8 = _mm256_fmadd_ps(a_vec, b_vec, sum8);
        }

        float sum = hsum256_ps(sum8);
        for (; i < d; ++i) {
            sum += a[i] * b[i];
        }

        return -sum;
    }

    void set_query(const float* x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q = &storage[idx * d];
    }
};

class GenericDistanceComputerL2_AVX2 : public DistanceComputer {
    
    int d;  // vector dimension
    const std::vector<float>& storage;  // vector datas
    const float* q;  // pointer for query vector

    public:
        GenericDistanceComputerL2_AVX2(const std::vector<float>& storage, int d) 
        : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        const float* base = &storage[index * d];
        __m256 sum8 = _mm256_setzero_ps();
        int i = 0;

        for (; i + 7 < d; i += 8) {
            __m256 q_vec = _mm256_loadu_ps(q + i);
            __m256 b_vec = _mm256_loadu_ps(base + i);
            __m256 diff = _mm256_sub_ps(q_vec, b_vec);
            sum8 = _mm256_fmadd_ps(diff, diff, sum8);
        }

        float sum = hsum256_ps(sum8);
        for (; i < d; ++i) {
            float diff = q[i] - base[i];
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        const float* a = &storage[idx_a * d];
        const float* b = &storage[idx_b * d];
        __m256 sum8 = _mm256_setzero_ps();
        int i = 0;

        for (; i + 7 < d; i += 8) {
            __m256 a_vec = _mm256_loadu_ps(a + i);
            __m256 b_vec = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(a_vec, b_vec);
            sum8 = _mm256_fmadd_ps(diff, diff, sum8);
        }

        float sum = hsum256_ps(sum8);
        for (; i < d; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return std::sqrt(sum); 
    }

    void set_query(const float* x) override {
        q = x;
    }

    void set_query_storage(int idx) override {
        q = &storage[idx * d];
    }
};


// AVX512 horizenal sum
inline float hsum512_ps(__m512 v) {
    __m256 v256 = _mm256_add_ps(_mm512_extractf32x8_ps(v, 1), _mm512_castps512_ps256(v));
    __m128 v128 = _mm_add_ps(_mm256_extractf128_ps(v256, 1), _mm256_castps256_ps128(v256));
    v128 = _mm_add_ps(v128, _mm_movehdup_ps(v128));
    v128 = _mm_add_ss(v128, _mm_movehl_ps(v128, v128));
    return _mm_cvtss_f32(v128);
}

class GenericDistanceComputerIP_AVX512 : public DistanceComputer {
    int d;
    const std::vector<float>& storage;
    const float* q;

public:
    GenericDistanceComputerIP_AVX512(const std::vector<float>& storage, int d) 
        : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        if (!q) throw std::runtime_error("Query pointer not set");
        
        const float* base = &storage[index * d];
        __m512 sum16 = _mm512_setzero_ps();
        int i = 0;

        // AVX512 main loop
        for (; i + 15 < d; i += 16) {
            __m512 q_vec = _mm512_loadu_ps(q + i);
            __m512 b_vec = _mm512_loadu_ps(base + i);
            sum16 = _mm512_fmadd_ps(q_vec, b_vec, sum16);
        }

        // residue elements
        float sum = hsum512_ps(sum16);
        for (; i < d; ++i) sum += q[i] * base[i];

        return -sum;
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        const float* a = &storage[idx_a * d];
        const float* b = &storage[idx_b * d];
        __m512 sum16 = _mm512_setzero_ps();
        int i = 0;

        for (; i + 15 < d; i += 16) {
            __m512 a_vec = _mm512_loadu_ps(a + i);
            __m512 b_vec = _mm512_loadu_ps(b + i);
            sum16 = _mm512_fmadd_ps(a_vec, b_vec, sum16);
        }

        float sum = hsum512_ps(sum16);
        for (; i < d; ++i) sum += a[i] * b[i];
        return -sum;
    }

    void set_query(const float* x) override { q = x; }
    void set_query_storage(int idx) override { q = &storage[idx * d]; }
};

class GenericDistanceComputerL2_AVX512 : public DistanceComputer {
    int d;
    const std::vector<float>& storage;
    const float* q;

public:
    GenericDistanceComputerL2_AVX512(const std::vector<float>& storage, int d)
        : storage(storage), d(d), q(nullptr) {}

    float operator()(int index) override {
        const float* base = &storage[index * d];
        __m512 sum16 = _mm512_setzero_ps();
        int i = 0;

        // AVX512 main loop
        for (; i + 15 < d; i += 16) {
            __m512 q_vec = _mm512_loadu_ps(q + i);
            __m512 b_vec = _mm512_loadu_ps(base + i);
            __m512 diff = _mm512_sub_ps(q_vec, b_vec);
            sum16 = _mm512_fmadd_ps(diff, diff, sum16);
        }

        // residue elements
        float sum = hsum512_ps(sum16);
        for (; i < d; ++i) {
            float diff = q[i] - base[i];
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

    float symmetric_dis(int idx_a, int idx_b) override {
        const float* a = &storage[idx_a * d];
        const float* b = &storage[idx_b * d];
        __m512 sum16 = _mm512_setzero_ps();
        int i = 0;

        for (; i + 15 < d; i += 16) {
            __m512 a_vec = _mm512_loadu_ps(a + i);
            __m512 b_vec = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(a_vec, b_vec);
            sum16 = _mm512_fmadd_ps(diff, diff, sum16);
        }

        float sum = hsum512_ps(sum16);
        for (; i < d; ++i) sum += (a[i] - b[i]) * (a[i] - b[i]);
        return std::sqrt(sum);
    }

    void set_query(const float* x) override { q = x; }
    void set_query_storage(int idx) override { q = &storage[idx * d]; }
};