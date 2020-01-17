// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "mem_buffer.h"

#include "onnx-ml.pb.h"
#include "predict.pb.h"
#include "single_include/nlohmann/json.hpp"

using json = nlohmann::json;

namespace onnxruntime {
namespace server {
// How much memory it will need for putting the content of this tensor into a plain array
// complex64/complex128 tensors are not supported.
// The output value could be zero or -1.
template <size_t alignment>
void GetSizeInBytesFromShapeAndElementType(const std::vector<int64_t>& shape, const ONNXTensorElementDataType& ele_type, size_t* out);
template <size_t alignment>
void GetSizeInBytesFromTensorProto(const onnx::TensorProto& tensor_proto, size_t* out);
/**
 * deserialize a TensorProto into a preallocated memory buffer.
 *  Impl must correspond to onnxruntime/core/framework/tensorprotoutils.cc
 * This implementation does not support external data so as to reduce dependency surface.
 */
void TensorProtoToMLValue(const onnx::TensorProto& input, const server::MemBuffer& m, /* out */ Ort::Value& value);

template <typename T>
void UnpackTensor(const json& input_json,
                  /*out*/ T* p_data, int64_t expected_size);
template <typename T>
void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                  /*out*/ T* p_data, int64_t expected_size);
void OrtInitializeBufferForTensor(void* input, size_t input_len, ONNXTensorElementDataType type);
ONNXTensorElementDataType CApiElementTypeFromProtoType(int type);
ONNXTensorElementDataType GetTensorElementType(const onnx::TensorProto& tensor_proto);
}  // namespace server
}  // namespace onnxruntime