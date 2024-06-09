# candle-nonzero
cpu/cuda nonzero op of candle framework

add nonzero op for [candle](https://github.com/huggingface/candle) framework, support both cpu and cuda

## Benchmark

use cargo bench to run the benchmark
```
cargo bench
```
given a tensor with N elements, and 4 dimensions, the shape is [N/8,2,2,2]

1/6 of the elements are non-zero, the rest are zero.

testing on  4090 GPU and 16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz


### N = 2^10
|  Type  | Performance |
| -----------   | -----------     |
| GPU     | 40.460 µs           |
| CPU(rayon Parallel)           |        327.70 µs         |
| CPU(sequential)| 2.7161 µs |

### N = 2^15

|  Type  | Performance |
| -----------   | -----------     |
| GPU     |  43.611 µs           |
| CPU(rayon Parallel)           |      3.5588 ms           |
| CPU(sequential)| 66.236 µs |

### N = 2^20

|  Type  | Performance |
| -----------   | -----------     |
| GPU     | 52.509 µs           |
| CPU(rayon Parallel)           |     78.130 ms            |
| CPU(sequential)| 2.2676 ms |

