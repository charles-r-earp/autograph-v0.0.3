use std::{fmt::{self, Debug, Display}, error::Error};
use super::{hipError_t, hipblasStatus_t, miopenStatus_t};

pub(super) enum RocmError {
    HipError(hipError_t),
    HipblasStatus(hipblasStatus_t),
    MiopenStatus(miopenStatus_t)
}

impl From<hipError_t> for RocmError {
    fn from(e: hipError_t) -> Self {
        RocmError::HipError(e)
    }
}

impl From<hipblasStatus_t> for RocmError {
    fn from(s: hipblasStatus_t) -> Self {
        RocmError::HipblasStatus(s)
    }
}

impl From<miopenStatus_t> for RocmError {
    fn from(s: miopenStatus_t) -> Self {
        RocmError::MiopenStatus(s)
    }
}

impl Debug for RocmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RocmError::HipError(e) => e.fmt(f),
            RocmError::HipblasStatus(s) => s.fmt(f),
            RocmError::MiopenStatus(s) => s.fmt(f)
        }
    }
}

impl Display for RocmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self, f)
    }
}

impl Error for RocmError {
    fn description(&self) -> &'static str {
        "RocmError"
    }
}

pub(super) type RocmResult<T> = Result<T, RocmError>;

pub(super) trait IntoResult {
    type Error: Error;
    fn into_result(self) -> Result<(), Self::Error>;
}

impl IntoResult for hipError_t {
    type Error = RocmError;
    fn into_result(self) -> Result<(), Self::Error> {
        match self {
            hipError_t::hipSuccess => Ok(()),
            _ => Err(self.into())
        }
    }
}

impl IntoResult for hipblasStatus_t {
    type Error = RocmError;
    fn into_result(self) -> Result<(), Self::Error> {
        match self {
            hipblasStatus_t::HIPBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(self.into())
        }
    }
}

impl IntoResult for miopenStatus_t {
    type Error = RocmError;
    fn into_result(self) -> Result<(), Self::Error> {
        match self {
            miopenStatus_t::miopenStatusSuccess => Ok(()),
            _ => Err(self.into())
        }
    }
}
