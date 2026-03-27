import train
import test
import visualize

if __name__ == "__main__":
    print("=" * 50)
    print("Step 1: Training TrojanNet (ResNet50 is frozen)")
    print("=" * 50)
    train.main()
    
    print("\n" + "=" * 50)
    print("Step 2: Evaluating clean accuracy and attack success rate")
    print("=" * 50)
    test.main()
    
    print("\n" + "=" * 50)
    print("Step 3: Visualizing clean vs trojan predictions")
    print("=" * 50)
    visualize.main()