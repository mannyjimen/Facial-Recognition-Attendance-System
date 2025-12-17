import recognition
import train_model
import numpy as np

image_size = 100 * 100

def test_training_model():
    num_people = 3 #me, junyi, sammy
    num_images_per_person = 5
    num_images = num_people * num_images_per_person

    
    mock_X = np.random.rand(num_images, image_size).astype(np.float32)

    #creating mock lables
    labels = ['Manny', 'Sammy', 'Junyi']
    mock_labels = np.array([label for label in labels for _ in range(num_images_per_person)])

    print("starting to train with mock data")
    train_model.train_model(
        face_data = mock_X,
        labels = mock_labels,
        num_reduced_dimensions = 5,
        num_neighors = 4)
    
    print ("finished training model")

def test_dummy_face():
    load_success = recognition.load_recognition_models()
    if (load_success):
        print("going to test with dummy array:")
        dummy_face = np.random.rand(100, 100).astype(np.float32)

        try:
            result = recognition.recognize_face(dummy_face)
            print(f"prediction result : {result}")
        except Exception as e:
            print(f"ERROR during recognition: {e}")


if __name__ == "__main__":
    test_training_model()
    test_dummy_face()