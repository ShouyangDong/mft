
import caffe.proto.caffe_pb2 as caffe_pb2
import bangir_pb2
from google.protobuf import text_format


def get_caffe_net(prototxt_file):
    net = caffe_pb2.NetParameter()
    with open(prototxt_file,'r') as f:
        text_format.Merge(f.read(), net)
    return net

def get_layer_info(layer):
    layer_info = {}
    layer_info['name'] = layer.name
    layer_info['top'] = layer.top
    layer_info['bottom'] = layer.bottom
    layer_info['type'] = layer.type
    return layer_info


def file_series(net, filename):
    model = bangir_pb2.ResNet()

    for idx, layer in enumerate(net.layer):
        layer_info = get_layer_info(layer)
        # print (layer_info)

        node = model.node.add()
        node.name = layer_info['name']
        node.top_name.extend(layer_info['top'])
        node.bottom_name.extend(layer_info['bottom'])
        node.op = layer_info['type']
        node.op_index = idx

    with open(filename, 'w') as f:
        f.write(str(model))

def read_proto(new_filename):
    model = bangir_pb2.ResNet()
    # print(address_book)
    f = open(new_filename, "rb")
    text_format.Parse(f.read(),model)
    # address_book.ParseFromString(f.read)
    f.close()
    print (new_filename)
    for node in model.node:
        print("p_op = {},p_name = {},p_index = {}".format(node.op,node.name,node.op_index))
    print ('=======')

if __name__ == "__main__":

    resnet50_prototxt_file = 'ResNet-50-deploy.prototxt'
    resnet101_prototxt_file = 'ResNet-101-deploy.prototxt'
    resnet152_prototxt_file = 'ResNet-152-deploy.prototxt'
    resnet_list = ['ResNet-50-deploy', 'ResNet-101-deploy', 'ResNet-152-deploy']

    for resnet_version in resnet_list:
        prototxt_file = "./" + resnet_version + '.prototxt'
        filename = "./" + resnet_version + ".pbtxt"

        net = get_caffe_net(prototxt_file)
        file_series(net, filename)
        read_proto(filename)

