#pragma once

extern "C" {
__attribute__((constructor)) void initgpu();
void memcpy_src2gpu(const int16_t **srcs, const int32_t *lens);
void resolver(const int16_t *chunk, const int32_t chunk_len, const bool *mask, uint32_t **result_raw, int32_t *minimum_val_startpos);
__attribute__((destructor)) void deinitgpu();
}