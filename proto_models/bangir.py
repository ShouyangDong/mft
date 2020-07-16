import bangir_pb2
from google.protobuf import text_format


def file_series(filename):
    model = bangir_pb2.FCNetwork()
    node_0 = model.node.add()
    node_0.name = "fc_0"
    node_0.op = "fc"
    node_0.op_index = 0
    node_0.data_type = bangir_pb2.FLOAT16
    node_0.input.extend(["1", "2","3"])
    ints_list = bangir_pb2.ListOfInteger()
    ints_list.ints.extend([1,2])
    node_0.input_datashape.extend([ints_list])
    # node_0.data_shape.extend(1)
    # node_0.data_shape.extend(2)
    node_0.dataparam.layout = bangir_pb2.NC
    node_0.fcparam.units = 16
    node_0.fcparam.bias = True
    node_0.fcparam.activation = 1
    node_0.input_memory_space.extend(0) 
    node_0.output_memory_space.extend(0) 
    # node_0.opconfiguration.extend(1)

    node_1 = model.node.add()
    node_1.name = "fc_1"
    node_1.op = "fc"
    node_1.op_index = 1
    node_1.data_type = bangir_pb2.FLOAT16
    node_1.input.extend(["1", "2","3"])
    ints_list = bangir_pb2.ListOfInteger()
    ints_list.ints.extend([1,2])
    node_1.input_datashape.extend([ints_list])
    # node_0.data_shape.extend(1)
    # node_0.data_shape.extend(2)
    node_1.dataparam.layout = bangir_pb2.NC
    node_1.fcparam.units = 8
    node_1.fcparam.bias = True
    node_1.fcparam.activation = 0
    node_1.input_memory_space.extend(0) 
    node_1.output_memory_space.extend(0) 
    # node_1.opconfiguration.extend(1)

    with open(filename, 'w') as f:
        f.write(str(model))

def read_proto(new_filename):
    model = bangir_pb2.FCNetwork()
    # print(address_book)
    f = open(new_filename, "rb")
    text_format.Parse(f.read(),model)
    # address_book.ParseFromString(f.read)
    f.close()
    
    for node in model.node:
        print("p_op = {},p_name = {},p_index = {},p_data_type = {}".format(node.op,node.name,node.op_index,node.data_type))

#序列化
# serializeToString = address_book.SerializeToString()
# print(serializeToString,type(serializeToString))

# address_book.ParseFromString(serializeToString)


if __name__ == "__main__":
    filename ="./bangir.pbtxt"
    file_series(filename)
    read_proto(filename)



