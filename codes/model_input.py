import tensorflow as tf

TFLITE_MODEL_PATH = "herbaura_best_float.tflite"

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT DETAILS:")
print(input_details)

print("\nOUTPUT DETAILS:")
print(output_details)
