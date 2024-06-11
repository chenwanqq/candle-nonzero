use candle_core::{
    backend::BackendStorage,
    bail,
    cuda::{kernel_name, Map2, WrapErr},
    CpuStorage, CudaDevice, CudaStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor,
    WithDType,
};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::ops::{BitAnd, BitOr, BitXor};
mod kernels;

pub enum BitWiseOpEnum {
    AND,
    OR,
    XOR,
}

impl ToString for BitWiseOpEnum {
    fn to_string(&self) -> String {
        match self {
            BitWiseOpEnum::AND => "AND".to_string(),
            BitWiseOpEnum::OR => "OR".to_string(),
            BitWiseOpEnum::XOR => "XOR".to_string(),
        }
    }
}

struct BitWise {
    pub op: BitWiseOpEnum,
}

impl BitWise {
    pub fn new(op: BitWiseOpEnum) -> Self {
        Self { op }
    }

    fn bitwise<T: WithDType + BitAnd<Output = T> + BitOr<Output = T> + BitXor<Output = T>>(
        &self,
        vs1: &[T],
        vs2: &[T],
    ) -> Vec<T> {
        let n = vs1.len();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let v1 = vs1[i];
            let v2 = vs2[i];
            let r = match self.op {
                BitWiseOpEnum::AND => v1 & v2,
                BitWiseOpEnum::OR => v1 | v2,
                BitWiseOpEnum::XOR => v1 ^ v2,
            };
            result.push(r);
        }
        result
    }
}

fn next_power_of_2(n: usize) -> usize {
    let mut result = 1;
    while result < n {
        result <<= 1;
    }
    result
}

impl CustomOp2 for BitWise {
    fn name(&self) -> &'static str {
        "bitwise"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if l1 != l2 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise",
            });
        }
        match s1 {
            CpuStorage::U8(vs1) => {
                let vs2 = s2.as_slice::<u8>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::U8(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::U32(vs1) => {
                let vs2 = s2.as_slice::<u32>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::U32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I64(vs1) => {
                let vs2 = s2.as_slice::<i64>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::I64(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::BF16(_) => Err(Error::UnsupportedDTypeForOp(DType::BF16, "bitwise")),
            CpuStorage::F16(_) => Err(Error::UnsupportedDTypeForOp(DType::F16, "bitwise")),
            CpuStorage::F32(_) => Err(Error::UnsupportedDTypeForOp(DType::F32, "bitwise")),
            CpuStorage::F64(_) => Err(Error::UnsupportedDTypeForOp(DType::F64, "bitwise")),
        }
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        impl Map2 for BitWise {
            fn f<T: cudarc::driver::DeviceRepr + WithDType + cudarc::driver::ValidAsZeroBits>(
                &self,
                s1: &cudarc::driver::CudaSlice<T>,
                l1: &Layout,
                s2: &cudarc::driver::CudaSlice<T>,
                l2: &Layout,
                dev: &CudaDevice,
            ) -> Result<cudarc::driver::CudaSlice<T>> {
                let slice1 = match l1.contiguous_offsets() {
                    None => bail!("input 1 has to be contiguous"),
                    Some((o1, o2)) => s1.slice(o1..o2),
                };
                let slice2 = match l2.contiguous_offsets() {
                    None => bail!("input 2 has to be contiguous"),
                    Some((o1, o2)) => s2.slice(o1..o2),
                };
                let elem_count = l1.shape().elem_count();
                let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
                let func =
                    match self.op {
                        BitWiseOpEnum::AND => dev
                            .get_or_load_func(&kernel_name::<T>("bitwise_and"), kernels::BITWISE)?,
                        BitWiseOpEnum::OR => {
                            dev.get_or_load_func(&kernel_name::<T>("bitwise_or"), kernels::BITWISE)?
                        }
                        BitWiseOpEnum::XOR => dev
                            .get_or_load_func(&kernel_name::<T>("bitwise_xor"), kernels::BITWISE)?,
                    };
                let block_size = next_power_of_2(elem_count).min(1024);
                let grid_size = (elem_count + block_size - 1) / block_size;
                let cfg = LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let params = (&slice1, &slice2, &dst, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(dst)
            }
        }
        if l1 != l2 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise",
            });
        }
        let dev = s1.device();
        let slice = self.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }
}

pub trait BitWiseOp {
    fn bitwise_and(&self, rhs: &Tensor) -> Result<Tensor>;
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor>;
    fn bitwise_xor(&self, rhs: &Tensor) -> Result<Tensor>;
}

impl BitWiseOp for Tensor {
    fn bitwise_and(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseOpEnum::AND))
    }

    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseOpEnum::OR))
    }

    fn bitwise_xor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseOpEnum::XOR))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bitwise_and() {
        use super::BitWiseOp;
        use candle_core::Tensor;
        let a = Tensor::from_vec(vec![1u32, 2, 3, 4], &[4], &candle_core::Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![1u32, 2, 3, 0], &[4], &candle_core::Device::Cpu).unwrap();
        let c = a.bitwise_and(&b).unwrap();
        assert_eq!(&c.to_vec1::<u32>().unwrap(), &[1, 2, 3, 0]);
        let dev = candle_core::Device::new_cuda(0).unwrap();
        let a = a.to_device(&dev).unwrap();
        let b = b.to_device(&dev).unwrap();
        let c = a.bitwise_and(&b).unwrap();
        assert_eq!(&c.to_vec1::<u32>().unwrap(), &[1, 2, 3, 0]);
    }
}
