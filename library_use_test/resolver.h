#pragma once

extern "C"{
    void resolver(const int16_t *problem, const int16_t **src, const int32_t problem_len, const int32_t *src_length, int32_t ** result);
}