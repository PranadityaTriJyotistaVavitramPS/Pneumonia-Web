import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

function App() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string>("");
  const [normalProbability, setNormalProbability] = useState<number | null>(null);
  const [pneumoniaProbability, setPneumoniaProbability] = useState<number | null>(null);
  const [healthAdvice, setHealthAdvice] = useState<string>("");

  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    console.log("Starting model loading...");
    try {
      const loadedModel = await tf.loadLayersModel("/model/model.json");
      console.log("Model loaded successfully:", loadedModel);
      setModel(loadedModel);
    } catch (error) {
      console.error("Failed to load model. Error:", error);
      alert("Model failed to load. Check console logs.");
    } finally {
      setLoadingModel(false);
    }
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      console.log("Image file selected:", file);
      setImageFile(file);
    }
  };

  const preprocessImage = (image: HTMLImageElement) => {
    console.log("Preprocessing image...");
    
    const tensor = tf.browser.fromPixels(image)
      .resizeBilinear([150, 150])  // Resize to 150x150, matching model input size
      .mean(-1)                    // Convert to grayscale (average RGB channels)
      .expandDims(-1)              // Add a channel dimension (shape: [150, 150, 1])
      .toFloat()
      .expandDims(0)               // Add batch dimension (shape: [1, 150, 150, 1])
      .div(255.0);                 // Normalize pixel values to range [0, 1]
    
    console.log("Image preprocessed successfully.");
    return tensor;
  };
  
  const handleClassify = async () => {
    if (!model) {
      alert("Model isn't loaded yet. Please wait.");
      return;
    }
  
    if (!imageFile) {
      alert("Please upload an image.");
      return;
    }
  
    const reader = new FileReader();
    reader.onload = async (event) => {
      const imageElement = new Image();
      if (event.target?.result) {
        imageElement.src = event.target.result as string;
        imageElement.onload = async () => {
          try {
            console.log("Image loaded. Preprocessing...");
            const inputTensor = preprocessImage(imageElement);
  
            console.log("Running prediction...");
            const predictionResult = model.predict(inputTensor) as tf.Tensor;

            console.log("Raw prediction result (tensor):", predictionResult);

            const predictionData = predictionResult.dataSync();  // or predictionResult.arraySync()
  
            console.log("Prediction data (array):", predictionData);
            const normalProbabilityValue = predictionData[0];
            const pneumoniaProbabilityValue = 1 - normalProbabilityValue;
  
            console.log("Probability of Normal:", normalProbabilityValue);
            console.log("Probability of Pneumonia:", pneumoniaProbabilityValue);
  
            const predictedClass = normalProbabilityValue > pneumoniaProbabilityValue ? "Normal" : "Pneumonia";
            console.log("Prediction decision:", predictedClass);
  
            // Set prediction and probabilities to state
            setPrediction(predictedClass);
            setNormalProbability(normalProbabilityValue * 100);  // convert to percentage
            setPneumoniaProbability(pneumoniaProbabilityValue * 100);  // convert to percentage

            // Set health advice based on prediction
            if (predictedClass === "Pneumonia") {
              setHealthAdvice(
                "If you have pneumonia, seek medical attention immediately. Early treatment with antibiotics or antivirals may be necessary depending on the type. Get plenty of rest, stay hydrated, and follow your doctor's advice for medications."
              );
            } else {
              setHealthAdvice(
                "Your lungs are healthy! To prevent pneumonia, maintain good hygiene, avoid smoking, stay active, and get vaccinated against pneumococcal pneumonia."
              );
            }
  
          } catch (error) {
            console.error("Error during prediction:", error);
            alert("Prediction failed.");
          }
        };
      }
    };
    reader.readAsDataURL(imageFile);
  };

  return (
    
    

    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-2xl font-bold mb-4 text-center">GROUP 2</h1>
        <div className="bg-white shadow-lg rounded-lg p-6 max-w-md">
          <h1 className="text-2xl font-bold mb-4 text-center">Pneumonia Classifier</h1>

          {loadingModel && (
            <div className="text-center mb-4 text-blue-600">Loading model... Please wait...</div>
          )}

          {/* Image Upload Input */}
          <input
            type="file"
            className="border p-2 mb-4 w-full"
            onChange={handleImageChange}
            disabled={loadingModel}
          />

          {/* Classify Button */}
          <button
            onClick={handleClassify}
            className={`w-full ${loadingModel ? "bg-gray-400" : "bg-blue-500 hover:bg-blue-600"} text-white py-2 px-4 rounded`}
            disabled={loadingModel}
          >
            Classify
          </button>

          {/* Display Prediction */}
          {prediction && (
            <div className="mt-4 text-center">
              <h2 className="text-lg font-semibold">Prediction Result:</h2>
              <p className="text-xl font-bold text-blue-600">{prediction}</p>
              <p className="text-xl font-bold text-green-600">
                Probability of Normal Lung: {normalProbability?.toFixed(2)}%
              </p>
              <p className="text-xl font-bold text-red-600">
                Probability of Pneumonia Lung: {pneumoniaProbability?.toFixed(2)}%
              </p>
            </div>
          )}

          {/* Display Health Advice */}
          {healthAdvice && (
            <div className="mt-4 text-center text-gray-700">
              <h2 className="text-lg font-semibold">Health Advice:</h2>
              <p>{healthAdvice}</p>
            </div>
          )}
        </div>
      </div>

      </div>

  );
}

export default App;
