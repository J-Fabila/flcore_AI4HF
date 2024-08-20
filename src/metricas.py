import torch
from binary import hd95
from binary import hd, jc, assd

a=torch.load("/home/jorge/work_dir/federado/Federado/ssh_version/data/MMs/MMs_1_reshaped.pt")
im = a[0]["image"].numpy()
gt = a[0]["mask"].int().numpy()
gt1 = a[1]["image"].int().numpy()
hache = hd(im,gt)
hache95 = hd95(im,gt)
print(im.shape, gt.shape)
print("COM",gt1.shape, gt.shape)
print("COM",hd95(gt1, gt))
print(hache95, hache)
print("IM", im.max())
print("IM", im.mean())

print("MASK",gt.max())
print("MASK",gt.min())
print("MASK",gt.mean())
print("MASK",gt[0,125,:])

gt = a[0]["mask"].int()
gt1 = a[1]["mask"].int()

_, d1 , d2 = gt.shape
original_shape = (1, d1,d2,4)
flat = gt.flatten()
encoded = torch.nn.functional.one_hot(flat.to(torch.int64),4)
target = encoded.view(original_shape)
target_prob = target.permute(0, 3, 1, 2).to(int)
print(target_prob.shape)
print(target_prob[0,2,125,:])

flat1 = gt1.flatten()
encoded = torch.nn.functional.one_hot(flat1.to(torch.int64),4)
target1 = encoded.view(original_shape)
target_prob1 = target1.permute(0, 3, 1, 2).to(int)
print(target_prob1.shape)
print(target_prob1[0,2,125,:])

target_prob[0,3,10,0] = 1

print(" FORMAS ", target_prob.shape, target_prob1.shape)
print("0",hd95(target_prob[0,0,:,:].numpy(), target_prob1[0,0,:,:].numpy()))
print("1",hd95(target_prob[0,1,:,:].numpy(), target_prob1[0,1,:,:].numpy()))
print("2",hd95(target_prob[0,2,:,:].numpy(), target_prob1[0,2,:,:].numpy()))
print("3",hd95(target_prob1[0,3,:,:].int().numpy(), target_prob[0,3,:,:].int().numpy()))

#print((target_prob1.float()-target_prob.float()).max())
print(target_prob[0,3,:,:].max(),target_prob.float()[0,3,:,:].mean(),target_prob[0,3,:,:].shape)
print(target_prob1[0,3,:,:].max(),target_prob1.float()[0,3,:,:].mean(),target_prob1[0,3,:,:].shape)

for i in range(256):
    print(target_prob1[0,3,i,125])
