import { useState } from 'react';
import './Outputs.css';

function Outputs() {
  const models = [
    {
      name: 'Our Model',
      accuracy: 92.5,
      precision: 91.8,
      recall: 90.7,
      f1: 91.2,
    },
    {
      name: 'BERT',
      accuracy: 89.4,
      precision: 88.5,
      recall: 87.9,
      f1: 88.2,
    },
    {
      name: 'XLM-R',
      accuracy: 90.1,
      precision: 89.2,
      recall: 88.6,
      f1: 88.9,
    },
  ];

  const [images, setImages] = useState([]);

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    const newImages = files.map((file) => URL.createObjectURL(file));
    setImages((prevImages) => [...prevImages, ...newImages]);
  };

  return (
    <section className="outputs">
      <div className="container">
        <h2>Model Performance & Outputs</h2>
        <div className="outputs-grid">
          {models.map((model) => (
            <div key={model.name} className="output-card">
              <h3>{model.name}</h3>
              <p><strong>Accuracy:</strong> {model.accuracy}%</p>
              <p><strong>Precision:</strong> {model.precision}%</p>
              <p><strong>Recall:</strong> {model.recall}%</p>
              <p><strong>F1-Score:</strong> {model.f1}%</p>
            </div>
          ))}
        </div>
        <p>
          Our model outperforms BERT and XLM-R in key metrics, showcasing superior capability in multilingual and multimodal classification.
        </p>
        <div className="image-upload">
          <h3>Upload Output Visualizations</h3>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handleImageUpload}
            className="upload-input"
          />
          <div className="image-grid">
            {images.map((image, index) => (
              <img key={index} src={image} alt={`Uploaded output ${index + 1}`} className="uploaded-image" />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default Outputs;