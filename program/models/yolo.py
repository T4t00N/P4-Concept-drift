num_experts = 2325
yolo_experts = nn.yolo_v8_n(len(params['names'].values())).cuda() for _ in range num_experts

train(yolo_experts)

boom = yolo experts resultat
