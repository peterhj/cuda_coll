#![allow(non_upper_case_globals)]

extern crate cuda;
extern crate float;

use ::ffi::*;

use cuda::ffi::runtime::{cudaStream_t};
use float::stub::{f16_stub};

use std::mem::{zeroed};
use std::ptr::{null_mut};

pub mod ffi;

#[derive(Clone, Copy, Debug)]
pub struct NcclError(pub ncclResult_t);

pub type NcclResult<T> = Result<T, NcclError>;

#[derive(Clone, Copy, Debug)]
pub enum NcclReduceOp {
  Sum,
  Prod,
  Max,
  Min,
}

impl NcclReduceOp {
  pub fn to_raw_op(&self) -> ncclRedOp_t {
    match *self {
      NcclReduceOp::Sum => ncclRedOp_t_ncclSum,
      NcclReduceOp::Prod => ncclRedOp_t_ncclProd,
      NcclReduceOp::Max => ncclRedOp_t_ncclMax,
      NcclReduceOp::Min => ncclRedOp_t_ncclMin,
    }
  }
}

pub trait NcclDataType: Copy {
  fn raw_data_type() -> ncclDataType_t;
}

impl NcclDataType for i8 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclInt8
  }
}

impl NcclDataType for u8 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclUint8
  }
}

impl NcclDataType for i32 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclInt32
  }
}

impl NcclDataType for u32 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclUint32
  }
}

impl NcclDataType for i64 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclInt64
  }
}

impl NcclDataType for u64 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclUint64
  }
}

impl NcclDataType for f16_stub {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclFloat16
  }
}

impl NcclDataType for f32 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclFloat32
  }
}

impl NcclDataType for f64 {
  fn raw_data_type() -> ncclDataType_t {
    ncclDataType_t_ncclFloat64
  }
}

pub struct NcclUniqueId {
  raw:  ncclUniqueId,
}

impl Clone for NcclUniqueId {
  fn clone(&self) -> Self {
    let mut new_uid = NcclUniqueId{raw: unsafe { zeroed() }};
    new_uid.raw.internal.copy_from_slice(&self.raw.internal);
    new_uid
  }
}

impl NcclUniqueId {
  pub fn create() -> NcclResult<Self> {
    let mut raw: ncclUniqueId = unsafe { zeroed() };
    let status = unsafe { ncclGetUniqueId(&mut raw as *mut _) };
    match status {
      ncclResult_t_ncclSuccess => Ok(NcclUniqueId{raw: raw}),
      _ => Err(NcclError(status)),
    }
  }
}

pub struct NcclComm {
  ptr:  ncclComm_t,
}

impl Drop for NcclComm {
  fn drop(&mut self) {
    let status = unsafe { ncclCommDestroy(self.ptr) };
    match status {
      ncclResult_t_ncclSuccess => {}
      _ => panic!(),
    }
  }
}

impl NcclComm {
  pub unsafe fn group_start() {
    ncclGroupStart();
  }

  pub unsafe fn group_end() {
    ncclGroupEnd();
  }

  pub fn init_rank(rank: i32, num_ranks: i32, comm_id: NcclUniqueId) -> NcclResult<NcclComm> {
    let mut ptr = null_mut();
    let status = unsafe { ncclCommInitRank(&mut ptr as *mut _, num_ranks, comm_id.raw, rank) };
    match status {
      ncclResult_t_ncclSuccess => Ok(NcclComm{ptr: ptr}),
      _ => Err(NcclError(status)),
    }
  }

  pub fn device(&self) -> NcclResult<i32> {
    let mut dev: i32 = 0;
    let status = unsafe { ncclCommCuDevice(self.ptr, &mut dev as *mut _) };
    match status {
      ncclResult_t_ncclSuccess => Ok(dev),
      _ => Err(NcclError(status)),
    }
  }

  pub fn rank(&self) -> NcclResult<i32> {
    let mut rank: i32 = 0;
    let status = unsafe { ncclCommUserRank(self.ptr, &mut rank as *mut _) };
    match status {
      ncclResult_t_ncclSuccess => Ok(rank),
      _ => Err(NcclError(status)),
    }
  }

  pub fn num_ranks(&self) -> NcclResult<i32> {
    let mut count: i32 = 0;
    let status = unsafe { ncclCommCount(self.ptr, &mut count as *mut _) };
    match status {
      ncclResult_t_ncclSuccess => Ok(count),
      _ => Err(NcclError(status)),
    }
  }

  pub unsafe fn reduce<T>(&mut self, send_buf: *const T, recv_buf: *mut T, count: usize, op: NcclReduceOp, root: i32, stream: cudaStream_t) -> NcclResult<()> where T: NcclDataType {
    let status = unsafe { ncclReduce(
        send_buf as *const _,
        recv_buf as *mut _,
        count,
        <T as NcclDataType>::raw_data_type(),
        op.to_raw_op(),
        root,
        self.ptr,
        stream,
    ) };
    match status {
      ncclResult_t_ncclSuccess => Ok(()),
      _ => Err(NcclError(status)),
    }
  }

  pub unsafe fn broadcast<T>(&mut self, buf: *mut T, count: usize, root: i32, stream: cudaStream_t) -> NcclResult<()> where T: NcclDataType {
    let status = unsafe { ncclBcast(
        buf as *mut _,
        count,
        <T as NcclDataType>::raw_data_type(),
        root,
        self.ptr,
        stream,
    ) };
    match status {
      ncclResult_t_ncclSuccess => Ok(()),
      _ => Err(NcclError(status)),
    }
  }

  pub unsafe fn all_reduce<T>(&mut self, send_buf: *const T, recv_buf: *mut T, count: usize, op: NcclReduceOp, stream: cudaStream_t) -> NcclResult<()> where T: NcclDataType {
    let status = unsafe { ncclAllReduce(
        send_buf as *const _,
        recv_buf as *mut _,
        count,
        <T as NcclDataType>::raw_data_type(),
        op.to_raw_op(),
        self.ptr,
        stream,
    ) };
    match status {
      ncclResult_t_ncclSuccess => Ok(()),
      _ => Err(NcclError(status)),
    }
  }

  pub unsafe fn reduce_scatter<T>(&mut self, send_buf: *const T, recv_buf: *mut T, recv_count: usize, op: NcclReduceOp, stream: cudaStream_t) -> NcclResult<()> where T: NcclDataType {
    let status = unsafe { ncclReduceScatter(
        send_buf as *const _,
        recv_buf as *mut _,
        recv_count,
        <T as NcclDataType>::raw_data_type(),
        op.to_raw_op(),
        self.ptr,
        stream,
    ) };
    match status {
      ncclResult_t_ncclSuccess => Ok(()),
      _ => Err(NcclError(status)),
    }
  }

  pub unsafe fn all_gather<T>(&mut self, send_buf: *const T, recv_buf: *mut T, send_count: usize, stream: cudaStream_t) -> NcclResult<()> where T: NcclDataType {
    let status = unsafe { ncclAllGather(
        send_buf as *const _,
        recv_buf as *mut _,
        send_count,
        <T as NcclDataType>::raw_data_type(),
        self.ptr,
        stream,
    ) };
    match status {
      ncclResult_t_ncclSuccess => Ok(()),
      _ => Err(NcclError(status)),
    }
  }
}
