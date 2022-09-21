################################################PoseNet_edge################################################
def conv_bn_block(input, output, kernel_size):
    '''
    标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    '''
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


def fc_bn_block(input, output):
    '''
    标准全连接块（fc + bn + relu）
    :param input:  输入通道数
    :param output:  输出通道数
    :return:  卷积核大小
    '''
    return nn.Sequential(
        nn.Linear(input, output),
        nn.ReLU(inplace=True)
    )

class SA_Layer_edge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        x_r = x_v @ attention # b, c, n 
        x = x + x_r
        return x

class CB_Layer_edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = x.contiguous().transpose(2, 1).contiguous()
        y = y.contiguous().transpose(2, 1).contiguous()
        y_t = y.permute(0, 2, 1) # b, n, c x= b, c, m        
        energy =  y_t @ x # b, n, m 
        attention = self.softmax(energy)# b, n, m 
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))# b, n, m 
        x_r = x @ attention.permute(0, 2, 1) # b, c, n 
        y_r = y @ attention # b, c, m
        x = x + y_r
        y = y + x_r

        out = torch.cat([x,y], dim=1)
        return out.contiguous().transpose(2, 1).contiguous()

class EdgeConv(nn.Module):
    '''
    EdgeConv模块
    1. 输入为：n * f
    2. 创建KNN graph，变为： n * k * f
    3. 接上若干个mlp层：a1, a2, ..., an
    4. 最终输出为：n * k * an
    5. 全局池化，变为： n * an
    '''
    def __init__(self, layers, K=20):
        '''
        构造函数
        :param layers: e.p. [3, 64, 64, 64]
        :param K:
        '''
        super(EdgeConv, self).__init__()

        self.K = K
        self.layers = layers
        # self.KNN_Graph = torch.zeros(Args.batch_size, 2048, self.K, self.layers[0]).to(Args.device)

        if layers is None:
            self.mlp = None
        else:
            mlp_layers = OrderedDict()
            for i in range(len(self.layers) - 1):
                if i == 0:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(2*self.layers[i], self.layers[i + 1], 1)
                else:
                    mlp_layers['conv_bn_block_{}'.format(i+1)] = conv_bn_block(self.layers[i], self.layers[i+1], 1)
            self.mlp = nn.Sequential(mlp_layers)

    def createSingleKNNGraph(self, X):
        '''
        generate a KNN graph for a single point cloud
        :param X:  X is a Tensor, shape: [N, F]
        :return: KNN graph, shape: [N, K, F]
        '''
        N, F = X.shape
        assert F == self.layers[0]

        # self.KNN_Graph = np.zeros(N, self.K)

        # 计算距离矩阵
        dist_mat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N) + \
                   torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N).t()
        dist_mat.addmm_(1, -2, X, X.t())

        # 对距离矩阵排序
        dist_mat_sorted, sorted_indices = torch.sort(dist_mat, dim=1)
        # print(dist_mat_sorted)

        # 取出前K个（除去本身）
        knn_indexes = sorted_indices[:, 1:self.K+1]
        # print(sorted_indices)

        # 创建KNN图
        knn_graph = X[knn_indexes]

        return knn_graph

    def forward(self, X):
        '''
        前向传播函数
        :param X:  shape: [B, N, F]
        :return:  shape: [B, N, an]
        '''
        # print(X.shape)
        B, N, F = X.shape

        assert F == self.layers[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        KNN_Graph = torch.zeros(B, N, self.K, self.layers[0]).to(device)

        # creating knn graph
        # X: [B, N, F]
        for idx, x in enumerate(X):
            # x: [N, F]
            # knn_graph: [N, K, F]
            # self.KNN_Graph[idx] = self.createSingleKNNGraph(x)
            KNN_Graph[idx] = self.createSingleKNNGraph(x)
        # print(self.KNN_Graph.shape)
        # print('KNN_Graph: {}'.format(KNN_Graph[0][0]))

        # X: [B, N, F]
        x1 = X.reshape([B, N, 1, F])
        x1 = x1.expand(B, N, self.K, F)
        # x1: [B, N, K, F]

        x2 = KNN_Graph - x1
        # x2: [B, N, K, F]

        x_in = torch.cat([x1, x2], dim=3)
        # x_in: [B, N, K, 2*F]
        x_in = x_in.permute(0, 3, 1, 2)
        # x_in: [B, 2*F, N, K]

        # reshape, x_in: [B, 2*F, N*K]
        x_in = x_in.reshape([B, 2 * F, N * self.K])

        # out: [B, an, N*K]
        out = self.mlp(x_in)
        _, an, _ = out.shape
        # print(out.shape)

        out = out.reshape([B, an, N, self.K])
        # print(out.shape)
        # reshape, out: [B, an, N, K]
        out = out.reshape([B, an*N, self.K])
        # print(out.shape)
        # reshape, out: [B, an*N, K]
        out = nn.MaxPool1d(self.K)(out)
        # print(out.shape)
        out = out.reshape([B, an, N])
        # print(out.shape)
        out = out.permute(0, 2, 1)
        # print(out.shape)

        return out




class PoseNetFeat_edge(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat_edge, self).__init__()
        self.combine1 = CB_Layer_edge()
        self.combine2 = CB_Layer_edge()

        self.conv1 = EdgeConv(layers=[3, 64, 64, 64], K=10)
        self.conv2 = EdgeConv(layers=[64, 128], K=10)

        self.e_conv1 = EdgeConv(layers=[32, 64, 64], K=10)
        self.e_conv2 = EdgeConv(layers=[64, 128], K=10)

        self.conv5 = EdgeConv(layers=[256, 512], K=10)
        self.conv6 = EdgeConv(layers=[512, 1024], K=10)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = self.combine1(x,emb)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = self.combine2(x,emb)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x.contiguous().transpose(2, 1).contiguous()).contiguous().transpose(2, 1).contiguous()

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points).contiguous().transpose(2, 1).contiguous()
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 2) #128 + 256 + 1024


class PoseNet_edge(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet_edge, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat1 = PoseNetFeat_edge(num_points)
        self.feat2 = PoseNetFeat_edge(num_points)

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj * 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj * 1, 1)  # confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        ap_x = self.feat1(x, emb.contiguous().transpose(1,2).contiguous())#500*1408
        ap_x = ap_x.contiguous().transpose(1,2).contiguous()

        tx = self.conv1_t(ap_x)
        cx = self.conv1_c(ap_x)

        tx = self.conv2_t(tx)
        cx = self.conv2_c(cx)

        tx = self.conv3_t(tx)
        cx = self.conv3_c(cx)

        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        b = 0
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        x_trans = (x-(x+out_tx.contiguous().transpose(1,2).contiguous()))#(500*3)
        ry = self.feat2(x_trans, emb.contiguous().transpose(1,2).contiguous())
        ry1 = self.conv1_r(ry.contiguous().transpose(1,2).contiguous())#(1024*500)
        ry2 = self.conv2_r(ry1)
        ry3 = self.conv3_r(ry2)
        out_rx = self.conv4_r(ry3).view(bs, self.num_obj, 4, self.num_points)
        out_rx = torch.index_select(out_rx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()