import torch
import torch.nn as nn
import torch.nn.functional as F
import pointfly as pf


def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training,
          with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):

    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D]  # (N,P,K)

    if sorting_method:
        indices = pf.sort_points(pts, indices, sorting_method)

    # gather neighbors
    nn_pts = torch.gather(
        pts.unsqueeze(1).expand(-1, P, -1, -1),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    )

    nn_pts_center = qrs.unsqueeze(2)
    nn_pts_local = nn_pts - nn_pts_center

    # feature from points
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + "pts0", is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + "pts", is_training)

    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = torch.gather(
            fts.unsqueeze(1).expand(-1, P, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, fts.shape[-1])
        )
        nn_fts_input = torch.cat([nn_fts_from_pts, nn_fts_from_prev], dim=-1)

    # X-Transformation
    if with_X_transformation:
        X0 = pf.conv2d(nn_pts_local, K * K, tag + 'X0', is_training, (1, K))
        X0 = X0.reshape(N, P, K, K)
        X1 = pf.depthwise_conv2d(X0, K, tag + 'X1', is_training, (1, K))
        X1 = X1.reshape(N, P, K, K)
        X2 = pf.depthwise_conv2d(X1, K, tag + 'X2', is_training, (1, K), activation=None)
        X2 = X2.reshape(N, P, K, K)

        fts_X = torch.matmul(X2, nn_fts_input)
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'conv', is_training, (1, K))
    fts_conv = fts_conv.squeeze(2)

    if with_global:
        fts_global = pf.dense(qrs, C // 4, tag + 'g1', is_training)
        fts_global = pf.dense(fts_global, C // 4, tag + 'g2', is_training)
        return torch.cat([fts_global, fts_conv], dim=-1)

    return fts_conv

class PointCNN(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting

    def forward(self, points, features, is_training=True):
        N = points.shape[0]

        xconv_params = self.setting.xconv_params
        layer_pts = [points]

        if features is None:
            layer_fts = [None]
        else:
            C_fts = xconv_params[0]['C'] // 2
            layer_fts = [pf.dense(features, C_fts, "fts_hd", is_training)]

        for i, prm in enumerate(xconv_params):
            K, D, P, C, links = prm['K'], prm['D'], prm['P'], prm['C'], prm['links']
            pts = layer_pts[-1]
            fts = layer_fts[-1]

            # sampling
            if P == -1:
                qrs = pts
            else:
                qrs = pts[:, :P, :]

            layer_pts.append(qrs)

            if i == 0:
                C_pts_fts = C // 2 if fts is None else C // 4
                depth_multiplier = 4
            else:
                C_prev = xconv_params[i - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = (C + C_prev - 1) // C_prev

            fts_x = xconv(pts, fts, qrs, f'xconv_{i}_', N, K, D, P, C,
                          C_pts_fts, is_training, self.setting.with_X_transformation,
                          depth_multiplier, self.setting.sorting_method,
                          with_global=(self.setting.with_global and i == len(xconv_params) - 1))

            layer_fts.append(fts_x)

        return layer_pts[-1], layer_fts[-1]

def pointcnn_loss(predictions, labels):
    return F.cross_entropy(predictions, labels)
