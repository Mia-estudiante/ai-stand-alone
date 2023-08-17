import torch

def evaluate(model, loader, device):
    with torch.no_grad(): #evaluate 시, gradient 를 구할 필요 없음
        model.eval() #평가라는 것을 모델에게 알려줌
        total = 0
        correct = 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            
            output_index = torch.argmax(output, dim=1)
            total += targets.shape[0]
            correct += (targets==output_index).sum().item()

    acc = correct/total*100
    model.train() #평가가 끝나면 다시 학습으로 돌아가도록 알려줌
    return acc

#클래스별 평가함수
def evaluate_by_class(model, loader, device, num_classes):
    with torch.no_grad():
        model.eval() 
        correct = torch.zeros(num_classes) 
        total = torch.zeros(num_classes) 
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            output_index = torch.argmax(output, dim=1)
            
            for _class in range(num_classes):
                total[_class] += (targets==_class).sum().item()
                correct[_class] += ((targets==_class) * (output_index==_class)).sum().item()

    acc = correct/total*100
    model.train() #평가가 끝나면 다시 학습으로 돌아가도록 알려줌
    return acc