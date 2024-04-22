import torch
import torch.nn as nn
# from kmeans_pytorch import kmeans
from kmeans_gpu import KMeans
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

class LASRModel:
    def __init__(self, n_bones=100):
        self.n_bones = n_bones
        self.ctl_ts  = nn.Parameter(torch.zeros(self.n_bones,3).cuda())
        self.ctl_rs  = nn.Parameter(torch.Tensor([[1,0,0,0]]).repeat(self.n_bones,1).cuda())
        self.log_ctl = nn.Parameter(torch.zeros(self.n_bones,3).cuda())
        self.kmeans = KMeans(
            n_clusters=n_bones,
            max_iter=100,
            tolerance=1e-4,
            distance='euclidean',
            sub_sampling=None,
            max_neighbors=15,
        )

    def init_bones(self, xyz):
        if self.n_bones > 2:
            # cluster_ids_xyz, xyz_cluster = kmeans(
            #     X=xyz, num_clusters=self.n_bones, distance='euclidean')
            # self.ctl_ts.data = xyz_cluster.to(xyz.device)
            self.ctl_ts.data = self.kmeans(xyz[None]).squeeze(0)
        else:
            self.ctl_ts.data = xyz.mean(0)[None]
        self.ctl_rs.data  = torch.Tensor([[1,0,0,0]]).repeat(self.n_bones,1).cuda()
        self.log_ctl.data = torch.ones(self.n_bones,3).cuda()

    def bones_to_all(self, all_xyz, d_ctl_xyz, d_ctl_rotation, d_ctl_scaling):
        dis_norm = self.ctl_ts.view(-1,1,3) - all_xyz.view(1,-1,3).detach() # p-v, H,J,1,3 - H,1,N,3
        dis_norm = dis_norm.matmul(quaternion_to_matrix(self.ctl_rs))
        dis_norm = self.log_ctl.exp().view(-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v)
        skin = (-10 * dis_norm.sum(-1)).softmax(1)[:,:,None] # h,j,n,1  bs, B, N, 1
        d_ctl_rot_matrix = quaternion_to_matrix(d_ctl_rotation).view(self.n_bones, 9)
        d_ctl = torch.cat([d_ctl_xyz, d_ctl_rot_matrix, d_ctl_scaling], dim=-1)
        d_all = torch.sum(d_ctl[:, None] * skin, dim=0)

        d_all_xyz = d_all[..., :3]
        d_all_rot = matrix_to_quaternion(d_all[..., 3:(3+9)].view(-1, 3, 3))
        d_all_scale = d_all[..., (3+9):(3+9+3)]
        return d_all_xyz, d_all_rot, d_all_scale
    
    def __call__(self, *args, **kwds):
        return self.bones_to_all(*args, **kwds)
