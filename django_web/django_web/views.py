from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse
from django.conf import settings
import uuid
import os
from glob import glob
from .library import *
from .module import TestData, Model
import json

folder_path = os.path.join(settings.BASE_DIR, 'uploads')
paramer_path = os.path.join(settings.BASE_DIR, 'parameters', 'parameters.pth')
if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')

def index(request):
    return render(request, 'index.html');

@require_http_methods(["POST"])
def upload(request):
    try:
        filename = str(uuid.uuid4())
        with open(os.path.join(folder_path, filename + '_' + request.FILES['file'].name), 'wb+') as destination:
            for chunk in request.FILES['file'].chunks():
                destination.write(chunk)
            return HttpResponse(status=200)
    except:
        return HttpResponse(status=503)

@require_http_methods(["POST"])
def execute(request):
    wavs_path = glob(os.path.join(folder_path, '*'))

    test_data = TestData(wavs_path)
    n_test_data = test_data.__len__()
    test_loader = DataLoader(test_data, batch_size=n_test_data, shuffle=False)

    shape = test_data.__getitem__(0).shape
    # print(shape)
    model = Model(input_shape=(1,shape[0],shape[1]), batch_size=n_test_data, num_category=20)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 2e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint = torch.load(paramer_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    model.to(device)
    count = 0
    pred = []
    for i, data in enumerate(test_loader):
        x = data
        batch, height, width = x.size()
        x = x.view(batch, 1, height, width)
        x = x.to(device, dtype=torch.float32)
        y_hat = model(x)

        for b in range(batch):
            path = wavs_path[b].split('/')[-1].split('_')[1:]
            str_path = path[0] + '_' + path[1]
            
            pred.append({'result':(torch.argmax(y_hat[b]) +1).item(),'path':str_path})
            
    return HttpResponse(json.dumps(pred))

