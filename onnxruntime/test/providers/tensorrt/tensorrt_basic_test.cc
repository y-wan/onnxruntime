// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "core/providers/tensorrt/tensorrt_execution_provider.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<float>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.template Data<float>(), rtensor.template Data<float>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

TEST(TensorrtExecutionProviderTest, FunctionTest) {
  onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  ///std::string model_file_name = "trt_execution_provider_function_test.onnx";
  ///status = onnxruntime::Model::Save(model, model_file_name);

  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTest.FunctionTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so};

  TensorrtExecutionProviderInfo epi;
  epi.device_id = 0;
  EXPECT_TRUE(session_object.RegisterExecutionProvider(onnxruntime::make_unique<::onnxruntime::TensorrtExecutionProvider>(epi)).IsOK());

  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  status = session_object.Load(sstr);
  ///status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

TEST(TensorrtExecutionProviderTest, DynamicShapeTest) {
  onnxruntime::Model model("graph_2", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim0");
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim1");
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim2");

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  auto& node1 = graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);
  node1.SetExecutionProviderType(onnxruntime::kTensorrtExecutionProvider);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  auto& node2 = graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);
  node2.SetExecutionProviderType(onnxruntime::kTensorrtExecutionProvider);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  ///std::string model_file_name = "trt_execution_provider_dynamicshape_test.onnx";
  ///status = onnxruntime::Model::Save(model, model_file_name);

  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTest.DynamicShapeTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so};

  TensorrtExecutionProviderInfo epi;
  epi.device_id = 0;
  EXPECT_TRUE(session_object.RegisterExecutionProvider(onnxruntime::make_unique<::onnxruntime::TensorrtExecutionProvider>(epi)).IsOK());

  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  status = session_object.Load(sstr);
  ///status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

TEST(TensorrtExecutionProviderTest, NodeIndexMappingTest) {
  onnxruntime::Model model("graph_3", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // BOOL tensor.
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // UINT8 tensor.
  ONNX_NAMESPACE::TypeProto uint8_tensor;
  uint8_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &bool_tensor);
  inputs.push_back(&input_arg_1);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("node_1_out", &uint8_tensor);
  outputs.push_back(&output_arg_1);
  auto& cast_node = graph.AddNode("cast1", "Cast", "node 1.", inputs, outputs);
  AttributeProto attr_proto;
  attr_proto.set_name("to");
  attr_proto.set_type(AttributeProto_AttributeType_INT);
  attr_proto.set_i(2);
  cast_node.AddAttribute("to", attr_proto);

  inputs.clear();
  inputs.push_back(&output_arg_1);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &bool_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  auto& cast_node_2 = graph.AddNode("cast2", "Cast", "node 2.", inputs, outputs);
  AttributeProto attr_proto_2;
  attr_proto_2.set_name("to");
  attr_proto_2.set_type(AttributeProto_AttributeType_INT);
  attr_proto_2.set_i(9);
  cast_node_2.AddAttribute("to", attr_proto_2);

  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&input_arg_2);
  inputs.push_back(&input_arg_3);
  auto& output_arg_3 = graph.GetOrCreateNodeArg("N", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_3);
  graph.AddNode("sub", "Sub", "node 3.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  ///std::string model_file_name = "trt_execution_provider_nodeindexmapping_test.onnx";
  ///status = onnxruntime::Model::Save(model, model_file_name);

  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<bool> values_mul_x = {true, false, true, false, true, false};
  std::vector<int64_t> dims_mul_y = {1, 3, 2};
  std::vector<float> values_mul_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<bool>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_y, values_mul_y, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_y, values_mul_y, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  output_names.push_back("N");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<bool> expected_values_mul_m = {true, false, true, false, true, false};
  std::vector<int64_t> expected_dims_mul_n = {1, 3, 2};
  std::vector<float> expected_values_mul_n = {0, 0, 0, 0, 0, 0};

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTest.NodeIndexMappingTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so};

  ///TensorrtExecutionProviderInfo epi;
  ///epi.device_id = 0;
  ///EXPECT_TRUE(session_object.RegisterExecutionProvider(onnxruntime::make_unique<::onnxruntime::TensorrtExecutionProvider>(epi)).IsOK());

  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  status = session_object.Load(sstr);
  ///status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  std::vector<OrtValue> fetche {fetches.back()};
  VerifyOutputs(fetche, expected_dims_mul_n, expected_values_mul_n);
}
/*
TEST_F(OpaqueTypeTests, RunModel) {
  SessionOptions so;
  so.session_logid = "SparseTensorTest";
  so.session_log_verbosity_level = 1;


  onnxruntime::Model model("graph_3", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // BOOL tensor.
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // UINT8 tensor.
  ONNX_NAMESPACE::TypeProto uint8_tensor;
  uint8_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &bool_tensor);
  inputs.push_back(&input_arg_1);
  auto& output_arg_1 = graph.GetOrCreateNodeArg("node_1_out", &uint8_tensor);
  outputs.push_back(&output_arg_1);
  auto& cast_node = graph.AddNode("cast1", "Cast", "node 1.", inputs, outputs);
  AttributeProto attr_proto;
  attr_proto.set_name("to");
  attr_proto.set_type(AttributeProto_AttributeType_INT);
  attr_proto.set_i(2);
  cast_node.AddAttribute("to", attr_proto);

  inputs.clear();
  inputs.push_back(&output_arg_1);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &bool_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  auto& cast_node_2 = graph.AddNode("cast2", "Cast", "node 2.", inputs, outputs);
  AttributeProto attr_proto_2;
  attr_proto_2.set_name("to");
  attr_proto_2.set_type(AttributeProto_AttributeType_INT);
  attr_proto_2.set_i(9);
  cast_node_2.AddAttribute("to", attr_proto_2);

  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&input_arg_2);
  inputs.push_back(&input_arg_3);
  auto& output_arg_3 = graph.GetOrCreateNodeArg("N", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_3);
  graph.AddNode("sub", "Sub", "node 3.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  ///std::string model_file_name = "trt_execution_provider_nodeindexmapping_test.onnx";
  ///status = onnxruntime::Model::Save(model, model_file_name);

  std::vector<int64_t> dims_mul_x = {1, 3, 2};
  std::vector<bool> values_mul_x = {true, false, true, false, true, false};
  std::vector<int64_t> dims_mul_y = {1, 3, 2};
  std::vector<float> values_mul_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<bool>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_y, values_mul_y, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestTensorrtExecutionProvider()->GetAllocator(0, OrtMemTypeCPU), dims_mul_y, values_mul_y, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  output_names.push_back("N");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 3, 2};
  std::vector<bool> expected_values_mul_m = {true, false, true, false, true, false};
  std::vector<int64_t> expected_dims_mul_n = {1, 3, 2};
  std::vector<float> expected_values_mul_n = {0, 0, 0, 0, 0, 0};

  SessionOptions so;
  so.session_logid = "TensorrtExecutionProviderTest.NodeIndexMappingTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so};

  TensorrtExecutionProviderInfo epi;
  epi.device_id = 0;
  EXPECT_TRUE(session_object.RegisterExecutionProvider(onnxruntime::make_unique<::onnxruntime::TensorrtExecutionProvider>(epi)).IsOK());

  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  status = session_object.Load(sstr);
  ///status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  std::vector<OrtValue> fetche {fetches.back()};
  VerifyOutputs(fetche, expected_dims_mul_n, expected_values_mul_n);




  // Both the session and the model need custom registries
  // so we construct it here before the model
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

  auto ops_schema = GetConstructSparseTensorSchema();
  auto shape_schema = GetFetchSparseShapeSchema();
  std::vector<OpSchema> schemas = {ops_schema, shape_schema};
  EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kMLDomain, 8, 9).IsOK());
  // Register our kernels here
  auto ctor_def = ConstructSparseTensorDef();
  EXPECT_TRUE(registry->RegisterCustomKernel(ctor_def, [](const OpKernelInfo& info) { return new ConstructSparseTensor(info); }).IsOK());
  auto shape_def = ConstructFetchSparseShape();
  EXPECT_TRUE(registry->RegisterCustomKernel(shape_def, [](const OpKernelInfo& info) { return new FetchSparseTensorShape(info); }).IsOK());

  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_ = {registry->GetOpschemaRegistry()};
  std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kMLDomain, 8}};

  Model model("SparseTensorTest", false, ModelMetaData(), custom_schema_registries_, domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  TypeProto input_tensor_proto(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());

  {
    // Sparse tensor will contain total 5 elements but only 2 of them a non-zero
    TypeProto input_values(input_tensor_proto);
    input_values.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    auto& sparse_values_arg = graph.GetOrCreateNodeArg("sparse_values", &input_values);
    inputs.push_back(&sparse_values_arg);

    TypeProto input_indicies(input_tensor_proto);
    input_indicies.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    auto& sparse_indicies_arg = graph.GetOrCreateNodeArg("sparse_indicies", &input_indicies);
    inputs.push_back(&sparse_indicies_arg);

    // Shape tensor will contain only one value
    TypeProto input_shape(input_tensor_proto);
    input_shape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& sparse_shape_arg = graph.GetOrCreateNodeArg("sparse_shape", &input_shape);
    inputs.push_back(&sparse_shape_arg);

    //Output is our custom data type
    TypeProto output_sparse_tensor(*DataTypeImpl::GetType<SparseTensorSample>()->GetTypeProto());
    auto& output_sparse_tensor_arg = graph.GetOrCreateNodeArg("sparse_rep", &output_sparse_tensor);
    outputs.push_back(&output_sparse_tensor_arg);

    auto& node = graph.AddNode("ConstructSparseTensor", "ConstructSparseTensor", "Create a sparse tensor representation",
                               inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }
  {
    // We start the input from previous node output
    inputs = std::move(outputs);
    outputs.clear();

    TypeProto output_shape(input_tensor_proto);
    output_shape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& output_shape_arg = graph.GetOrCreateNodeArg("sparse_tensor_shape", &output_shape);
    outputs.push_back(&output_shape_arg);
    auto& node = graph.AddNode("FetchSparseTensorShape", "FetchSparseTensorShape", "Fetch shape from sparse tensor",
                               inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  EXPECT_TRUE(graph.Resolve().IsOK());

  // Get a proto and load from it
  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  EXPECT_TRUE(session_object.Load(sstr).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;

  // Prepare inputs/outputs
  std::vector<int64_t> val_dims = {2};
  std::vector<int64_t> values = {1, 2};
  // prepare inputs
  OrtValue ml_values;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), val_dims, values, &ml_values);

  std::vector<int64_t> ind_dims = {2};
  std::vector<int64_t> indicies = {1, 4};
  OrtValue ml_indicies;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), ind_dims, indicies, &ml_indicies);

  std::vector<int64_t> shape_dims = {1};
  std::vector<int64_t> shape = {5};
  OrtValue ml_shape;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape_dims, shape, &ml_shape);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("sparse_values", ml_values));
  feeds.insert(std::make_pair("sparse_indicies", ml_indicies));
  feeds.insert(std::make_pair("sparse_shape", ml_shape));

  // Output
  std::vector<int64_t> output_shape_dims = {1};
  std::vector<int64_t> output_shape = {0};

  std::vector<std::string> output_names;
  output_names.push_back("sparse_tensor_shape");
  std::vector<OrtValue> fetches;

  EXPECT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  // Should get the original shape back in the form of a tensor
  EXPECT_EQ(1, rtensor.Shape().NumDimensions());
  EXPECT_EQ(5, *rtensor.template Data<int64_t>());
}
*/
}  // namespace test
}  // namespace onnxruntime
