import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import cv2
buc=100
step=0.05
Epo=80
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)), ])
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=buc, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=buc, shuffle=True, num_workers=0)
def kuo(a,p):
    p=p.view(1,-1)
    o=torch.ones(a,1)
    o=o.cuda(2)
    po=torch.mm(o,p)
    po=po.view(a,-1)
    return po
def mmp(p):
    a=p.size()[0]
    b=p.size()[1]
    c=p.size()[2]
    d=p.size()[3]
    ct=int(c/2)
    dt=int(d/2)

    pp=p.view(a*b,c,d)
    w1=torch.zeros(2,2)
    w1=w1.cuda(2)
    w1[0][0]+=1
    w2 = torch.zeros(2, 2)
    w2 = w2.cuda(2)
    w2[0][1] += 1
    w3 = torch.zeros(2, 2)
    w3 = w3.cuda(2)
    w3[1][0] += 1
    w4 = torch.zeros(2, 2)
    w4 = w4.cuda(2)
    w4[1][1] += 1
    t1=torch.zeros(c,2)
    t1=t1.cuda(2)
    t2=torch.zeros(2,d)
    t2=t2.cuda(2)
    for i in range(ct):
        t1[2*i][0]+=1
        t1[2*i+1][1]+=1
    for i in range(dt):
        t2[0][2*i]+=1
        t2[1][2*i+1]+=1
    you1=torch.zeros(d,d)
    you1=you1.cuda(2)
    for i in range(d-1):
        you1[i][i+1]=1
    zou1=torch.zeros(d,d)
    zou1=zou1.cuda(2)
    for i in range(d-1):
        zou1[i+1][i]=1
    shang1=torch.zeros(c,c)
    shang1=shang1.cuda(2)
    for i in range(c-1):
        shang1[i][i+1]=1
    xia1=torch.zeros(c,c)
    xia1=xia1.cuda(2)
    for i in range(c-1):
        xia1[i+1][i]=1
    ww1=torch.mm(t1,torch.mm(w1,t2))
    ww2 = torch.mm(t1, torch.mm(w2, t2))
    ww3 = torch.mm(t1, torch.mm(w3, t2))
    ww4 = torch.mm(t1, torch.mm(w4, t2))
    kw1=kuo(a*b,ww1).view(-1,c,d)
    kw2 = kuo(a * b, ww2).view(-1,c,d)
    kw3 = kuo(a * b, ww3).view(-1,c,d)
    kw4 = kuo(a * b, ww4).view(-1,c,d)
    xia=kuo(a*b,xia1).view(-1,c,c)
    shang=kuo(a*b,shang1).view(-1,c,c)
    zuo=kuo(a*b,zou1).view(-1,d,d)
    you=kuo(a*b,you1).view(-1,d,d)
    pw1=kw1*pp
    pwx1=pw1+torch.bmm(xia,pw1)+torch.bmm(pw1,you)+torch.bmm(xia,torch.bmm(pw1,you))
    pwx1=torch.sign(F.relu(pp-pwx1))
    pw2=kw2*pp
    pwx2=pw2+torch.bmm(pw2,zuo)+torch.bmm(xia,pw2)+torch.bmm(xia,torch.bmm(pw2,zuo))
    pwx2=torch.sign(F.relu(pp-pwx2))
    pw3 = kw3 * pp
    pwx3 = pw3 + torch.bmm(pw3, you) + torch.bmm(shang, pw3) + torch.bmm(shang, torch.bmm(pw3, you))
    pwx3 = torch.sign(F.relu(pp-pwx3))
    pw4 = kw4 * pp
    pwx4 = pw4 + torch.bmm(pw4, zuo) + torch.bmm(shang, pw4) + torch.bmm(shang, torch.bmm(pw4, zuo))
    pwx4 = torch.sign(F.relu(pp-pwx4))
    pwx=pwx1+pwx2+pwx3+pwx4-2.5
    pwx=torch.sign(F.relu(pwx))
    q1=torch.eye(c)
    q2=torch.eye(d)
    q1=q1.cuda(2)
    q2=q2.cuda(2)
    if c>ct*2:
        q1[c-1][c-1]=0
    if d>dt*2:
        q2[d-1][d-1]=0
    q11=kuo(a*b,q1).view(-1,c,c)
    q22=kuo(a*b,q2).view(-1,d,d)
    pwx0=torch.bmm(q11,torch.bmm(pwx,q22)).view(a,b,c,d)
    return pwx0

def zfo(p):
    p=torch.sign(p)
    a = p.size()[0]
    b = p.size()[1]
    c = p.size()[2]
    d = p.size()[3]
    ct = int(c / 2)
    dt = int(d / 2)
    zp= p.view(-1,c,d)
    z1=torch.zeros(ct,c)
    z2=torch.zeros(d,dt)
    for i in range(ct):
        z1[i][2*i]+=1
        z1[i][2*i+1]+=1
    for j in range(dt):
        z2[2*j][j]=1
        z2[2*j+1][j]=1

    z1=z1.cuda(2)
    z2=z2.cuda(2)
    z1=kuo(a*b,z1).view(-1,ct,c)
    z2=kuo(a*b,z2).view(-1,d,dt)
    zui=torch.bmm(z1,torch.bmm(zp,z2))
    zui=zui.view(a,b,ct,dt)
    zui=torch.sign(zui)
    return zui
def D1(a,i):
    p1 = a.size()[0]
    a=a.view(-1,1)
    c=torch.ones(1,i*i)
    c=c.cuda(2)
    ac=torch.mm(a,c).view(1,-1)
    b=torch.ones(buc,1)
    b=b.cuda(2)
    xx=torch.mm(b,ac)
    xx=xx.view(buc,p1,i,i)
    return xx

def D2(a):

    b=torch.ones(buc,1)
    b=b.cuda(2)
    a=a.view(1,-1)
    xx=torch.mm(b,a)
    xx=xx.view(buc,-1)
    return xx

class RFc(nn.Module):
    def __init__(self):
        super(RFc, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cov1=nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False)
        A1_p = torch.randn((64), requires_grad=True)
        self.A1_p = nn.Parameter(A1_p)
        self.bn1 = nn.BatchNorm2d(64)

        self.cov2 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias=False)
        A2_p = torch.randn((64), requires_grad=True)
        self.A2_p = nn.Parameter(A2_p)
        self.bn2 = nn.BatchNorm2d(64)

        self.cov3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=False)
        A3_p = torch.randn((128), requires_grad=True)
        self.A3_p = nn.Parameter(A3_p)
        self.bn3 = nn.BatchNorm2d(128)

        self.cov4 = nn.Conv2d(128,128, kernel_size=3, padding=1,bias=False)
        A4_p = torch.randn((128), requires_grad=True)
        self.A4_p = nn.Parameter(A4_p)
        self.bn4 = nn.BatchNorm2d(128)

        self.cov5 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False)
        A5_p = torch.randn((256), requires_grad=True)
        self.A5_p = nn.Parameter(A5_p)
        self.bn5 = nn.BatchNorm2d(256)


        self.cov6 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        A6_p = torch.randn((256), requires_grad=True)
        self.A6_p = nn.Parameter(A6_p)
        self.bn6 = nn.BatchNorm2d(256)


        self.cov8 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        A8_p = torch.randn((256), requires_grad=True)
        self.A8_p = nn.Parameter(A8_p)
        self.bn8 = nn.BatchNorm2d(256)

        self.cov9 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=False)
        A9_p = torch.randn((512), requires_grad=True)
        self.A9_p = nn.Parameter(A9_p)
        self.bn9 = nn.BatchNorm2d(512)

        self.cov10 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        A10_p = torch.randn((512), requires_grad=True)
        self.A10_p = nn.Parameter(A10_p)
        self.bn10 = nn.BatchNorm2d(512)


        self.cov12 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        A12_p = torch.randn((512), requires_grad=True)
        self.A12_p = nn.Parameter(A12_p)
        self.bn12 = nn.BatchNorm2d(512)

        self.cov13 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        A13_p = torch.randn((512), requires_grad=True)
        self.A13_p = nn.Parameter(A13_p)
        self.bn13 = nn.BatchNorm2d(512)

        self.cov14 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        A14_p = torch.randn((512), requires_grad=True)
        self.A14_p = nn.Parameter(A14_p)
        self.bn14 = nn.BatchNorm2d(512)


        self.cov16 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        A16_p = torch.randn((512), requires_grad=True)
        self.A16_p = nn.Parameter(A16_p)
        self.bn16 = nn.BatchNorm2d(512)

        self.ca1=nn.Linear(512*4,512*2,bias=False)
        A1q_p = torch.randn((512*2), requires_grad=True)
        self.A1q_p = nn.Parameter(A1q_p)

        self.ca2 = nn.Linear(512*2, 512, bias=False)
        A2q_p = torch.randn((512), requires_grad=True)
        self.A2q_p = nn.Parameter(A2q_p)

        self.ca3 = nn.Linear(512, 128, bias=False)
        A3q_p = torch.randn((128), requires_grad=True)
        self.A3q_p = nn.Parameter(A3q_p)

        self.ca4 = nn.Linear(128, 10, bias=False)
        A4q_p = torch.randn((10), requires_grad=True)
        self.A4q_p = nn.Parameter(A4q_p)



    def forward(self, x):
        x1=x
        x2=x


        x1 = self.bn1(self.cov1(x1)+D1(self.A1_p,32))
        #xq=self.cov1(x1)+D1(self.A1_p,32)

        #x1=self.bn1(self.cov1(x1)+D1(self.A1_p,32))
        #print(torch.sum(x1))
        x2=self.bn1(self.cov1(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)


        x1 = self.bn2(self.cov2(x1) + D1(self.A2_p,32))
        x2 = self.bn2(self.cov2(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)

        x1 = self.bn3(self.cov3(x1) + D1(self.A3_p,32))
        x2 = self.bn3(self.cov3(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)

        x1 = self.bn4(self.cov4(x1) + D1(self.A4_p,32))
        x2 = self.bn4(self.cov4(x2))
        x3 = mmp(x1)
        x2 = zfo(x3 * x2) * self.pool(abs(x3 * x2))
        x1 = F.relu(self.pool(x1))
        x2 = x2 * torch.sign(x1)

        x1 = self.bn5(self.cov5(x1) + D1(self.A5_p,16))
        x2 = self.bn5(self.cov5(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)

        x1 = self.bn6(self.cov6(x1) + D1(self.A6_p,16))
        x2 = self.bn6(self.cov6(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)



        x1 = self.bn8(self.cov8(x1) + D1(self.A8_p,16))
        x2 = self.bn8(self.cov8(x2))
        x3 = mmp(x1)
        x2 = zfo(x3 * x2) * self.pool(abs(x3 * x2))
        x1 = F.relu(self.pool(x1))
        x2 = x2 * torch.sign(x1)

        x1 = self.bn9(self.cov9(x1) + D1(self.A9_p,8))
        x2 = self.bn9(self.cov9(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)

        x1 = self.bn10(self.cov10(x1) + D1(self.A10_p,8))
        x2 = self.bn10(self.cov10(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)



        x1 = self.bn12(self.cov12(x1) + D1(self.A12_p,8))
        x2 = self.bn12(self.cov12(x2))
        x3 = mmp(x1)
        x2 = zfo(x3 * x2) * self.pool(abs(x3 * x2))
        x1 = F.relu(self.pool(x1))
        x2 = x2 * torch.sign(x1)

        x1 = self.bn13(self.cov13(x1) + D1(self.A13_p,4))
        x2 = self.bn13(self.cov13(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)

        x1 = self.bn14(self.cov14(x1) + D1(self.A14_p,4))
        x2 = self.bn14(self.cov14(x2))
        x1 = F.relu(x1)
        x2 = x2 * torch.sign(x1)



        x1 = self.bn16(self.cov16(x1) + D1(self.A16_p,4))
        x2 = self.bn16(self.cov16(x2))
        x3 = mmp(x1)
        x2 = zfo(x3 * x2) * self.pool(abs(x3 * x2))
        x1 = F.relu(self.pool(x1))
        x2 = x2 * torch.sign(x1)


        x1=x1.view(-1,512*4)
        x2=x2.view(-1,512*4)


        x1=F.relu(self.ca1(x1)+D2(self.A1q_p))
        x2=self.ca1(x2)*torch.sign(x1)

        x1 = F.relu(self.ca2(x1) + D2(self.A2q_p))
        x2 = self.ca2(x2) * torch.sign(x1)

        x1 = F.relu(self.ca3(x1) + D2(self.A3q_p))
        x2 = self.ca3(x2) * torch.sign(x1)

        x1 = self.ca4(x1) + D2(self.A4q_p)
        x2 = self.ca4(x2)

        return x1,x2
class RFc1(nn.Module):

    def __init__(self):

        super(RFc1, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cov1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        self.cov2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.cov3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        self.cov4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)

        self.cov5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(256)

        self.cov6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(256)

        self.cov8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.bn8 = nn.BatchNorm2d(256)

        self.cov9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.bn9 = nn.BatchNorm2d(512)

        self.cov10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn10 = nn.BatchNorm2d(512)

        self.cov12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn12 = nn.BatchNorm2d(512)

        self.cov13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn13 = nn.BatchNorm2d(512)

        self.cov14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn14 = nn.BatchNorm2d(512)

        self.cov16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn16 = nn.BatchNorm2d(512)

        self.ca1 = nn.Linear(512 * 4, 512 * 2, bias=True)


        self.ca2 = nn.Linear(512 * 2, 512, bias=True)


        self.ca3 = nn.Linear(512, 128, bias=True)


        self.ca4 = nn.Linear(128, 10, bias=True)



    def forward(self, x):
        x1 = x

        x1 = self.bn1(self.cov1(x1))
        x1 = F.relu(x1)

        x1 = self.bn2(self.cov2(x1))
        x1 = F.relu(x1)

        x1 = self.bn3(self.cov3(x1))
        x1 = F.relu(x1)

        x1 = self.bn4(self.cov4(x1))
        x1 = F.relu(self.pool(x1))

        x1 = self.bn5(self.cov5(x1))
        x1 = F.relu(x1)

        x1 = self.bn6(self.cov6(x1))
        x1 = F.relu(x1)

        x1 = self.bn8(self.cov8(x1))
        x1 = F.relu(self.pool(x1))

        x1 = self.bn9(self.cov9(x1))
        x1 = F.relu(x1)

        x1 = self.bn10(self.cov10(x1))
        x1 = F.relu(x1)

        x1 = self.bn12(self.cov12(x1))
        x1 = F.relu(self.pool(x1))

        x1 = self.bn13(self.cov13(x1))

        x1 = F.relu(x1)

        x1 = self.bn14(self.cov14(x1))

        x1 = F.relu(x1)

        x1 = self.bn16(self.cov16(x1))

        x1 = F.relu(self.pool(x1))

        x1 = x1.view(-1, 512 * 4)

        x1 = F.relu(self.ca1(x1))

        x1 = F.relu(self.ca2(x1))

        x1 = F.relu(self.ca3(x1))

        x1 = self.ca4(x1)

        return x1
net=RFc()
net=net.cuda(2)
net1=RFc1()
net1=net1.cuda(2)
criterion = nn.CrossEntropyLoss()
params=list(net.parameters())
opt = optim.SGD(net.parameters(),lr = step)
#scheduler = MultiStepLR(opt, milestones=[20,40,60,200], gamma=0.6)
net1.load_state_dict(torch.load('vgc-10-dlj(def)-1.pt'))
params1=list(net1.parameters())
for i in range(60):
    print(i,params[i].size(),params1[i].size())
for i in range(52):
    if int((i-1)/4)*4==i-1:
        params[int((i-1)/4)].data=params1[i]
    if int(i/4)*4==i:
        params[17+int(i/4)*3].data=params1[i]
for i in range(8):
    if int(i/2)*2==i:
        params[int(i/2)+56].data=params1[i+52]
    else:
        params[int((i-1)/2)+13].data=params1[i+52]
net.eval()
net1.eval()
def Fg( model, data, target,epsilon,i ):
    for k in range(i):
      data.requires_grad = True
      output,output1= model(data)
      lossvalue = criterion(output, target)
      model.zero_grad()
      lossvalue.backward()
      data_grad = data.grad.data
      data.requires_grad = False
      sign_data_grad = data_grad.sign()
      data= data + epsilon*sign_data_grad
    return data
for epoch in range(Epo):
    train_acc = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda(2)
        labels = labels.cuda(2)
        net.eval()
        xc = Fg(net, inputs, labels, 0.01, 10)
        xc = xc.cuda(2)
        net.train()
        net.zero_grad()
        output,output1 = net(xc)
        train_loss1 = criterion(output-output1, labels)
        train_loss2 = criterion(output, labels)
        train_loss=train_loss1+train_loss2
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        ous=output-output1
        _, pred = ous.max(1)
        num_correct = (pred == labels).sum()
        train_acc += int(num_correct)
    print(epoch,train_acc)
#    scheduler.step()
    test_acc=0
    net.eval()
    for data,labels in test_loader:
        net.zero_grad()
        data = data.cuda(2)
        labels = labels.cuda(2)
        output,pus = net(data)
        out1=output-pus
        _, pred = out1.max(1)
        num_correct = (pred == labels).sum()
        test_acc += int(num_correct)
    print(test_acc)

torch.save(net.state_dict(),'hmod(at)-c(dlj).pt')
