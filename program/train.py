import models.yolo
import models.MoCo
from models.mlp import MLP



def train():
    image = "path_to_random_img"

    feature_vector = MoCo(image)
    print(f"MoCO_feature_vector:{feature_vector}")

    expert_weights = MLP(feature_vector)
    print(f"Expert_weights:{expert_weights}")

    expert_loss = YOLO


    final_loss = (expert_weights[0]*expert_loss[0]+expert_weights[1]*
            expert_loss[1]+expert_weights[2]*expert_loss[2])

    pass





def main():
    print("Main things")

if __name__ == "__main__":
    main()
