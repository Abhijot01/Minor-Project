import { Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Outputs from './Outputs';

function Header() {
  return (
    <header className="header">
      <div className="container">
        <h1>Multimodal & Multilingual Social Media Analysis</h1>
        <p>AI-Powered Sentiment, Toxicity, and Relevance Classification</p>
      </div>
    </header>
  );
}

function Home() {
  return (
    <section className="home">
      <div className="container">
        <h2>Welcome to Our Project</h2>
        <p>
          Our project develops an intelligent system using AI and NLP to analyze social media content across multiple languages and formats (text and images). It classifies posts based on sentiment, toxicity, and relevance, enabling insights for moderation, brand management, and public opinion analysis.
        </p>
        <h3>Key Features</h3>
        <ul>
          <li><strong>Multilingual:</strong> Processes content in various languages for global use.</li>
          <li><strong>Multimodal:</strong> Analyzes text and images using NLP and OCR.</li>
          <li><strong>Sentiment Analysis:</strong> Detects positive, negative, or neutral tones.</li>
          <li><strong>Toxicity Detection:</strong> Flags harmful content like hate speech.</li>
          <li><strong>Relevance Classification:</strong> Identifies trending or context-specific topics.</li>
        </ul>
        <Link to="/outputs" className="btn">
          Explore Outputs
        </Link>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p>© 2025 Multimodal Multilingual Analysis Project. All rights reserved.</p>
    </footer>
  );
}

export default function App() {
  return (
    <div>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/outputs" element={<Outputs />} />
      </Routes>
      <Footer />
    </div>
  );
}