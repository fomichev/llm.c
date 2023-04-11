#pragma once

typedef struct {
	FT_TYPE v[FT_N];
} cpu_fv_t;

#define CPU_FV_LOAD(DST, SRC) \
	({ \
		__builtin_memcpy(&(DST).v[0], (SRC), FT_N * FT_SIZEOF); \
	})

#define CPU_FV_LOAD1(DST, VAL) \
	({ \
		for (size_t i = 0; i < FT_N; i++) { \
			(DST).v[i] = (VAL); \
		} \
	})

#define CPU_FV_STORE(DST, SRC) \
	({ \
		__builtin_memcpy((DST), &(SRC).v[0], FT_N * FT_SIZEOF); \
	})

#define CPU_FV_ADD(DST, LHS, RHS) \
	({ \
		for (size_t i = 0; i < FT_N; i++) { \
			(DST).v[i] = (LHS).v[i] + (RHS).v[i]; \
		} \
	})

#define CPU_FV_SUB(DST, LHS, RHS) \
	({ \
		for (size_t i = 0; i < FT_N; i++) { \
			(DST).v[i] = (LHS).v[i] - (RHS).v[i]; \
		} \
	})

#define CPU_FV_MUL(DST, LHS, RHS) \
	({ \
		for (size_t i = 0; i < FT_N; i++) { \
			(DST).v[i] = (LHS).v[i] * (RHS).v[i]; \
		} \
	})

#define CPU_FV_DIV(DST, LHS, RHS) \
	({ \
		for (size_t i = 0; i < FT_N; i++) { \
			(DST).v[i] = (LHS).v[i] / (RHS).v[i]; \
		} \
	})

#define CPU_FV_EXP(DST, LHS) \
	({ \
		for (size_t i = 0; i < FT_N; i++) { \
			(DST).v[i] = expf((LHS).v[i]); \
		} \
	})

#define CPU_FV_TANH(DST, LHS) \
	({ \
		for (size_t i = 0; i < FT_N; i++) { \
			(DST).v[i] = tahnf((LHS).v[i]); \
		} \
	})

#define CPU_FV_REDUCE_SUM(LHS) \
	({ \
		FT_TYPE sum = 0; \
		for (size_t i = 0; i < FT_N; i++) { \
			sum += (LHS).v[i]; \
		} \
		sum; \
	})
