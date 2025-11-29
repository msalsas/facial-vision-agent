# Facial Vision Agent

AI-powered agent for analyzing facial features and hair from images. Specialized in visual feature extraction only.

## Features

- 🎯 **Facial Feature Detection**: Face shape, proportions, prominent features
- 💇 **Hair Analysis**: Type, length, color, density, condition  
- 📊 **Confidence Metrics**: Quality assessment of analysis
- 🚫 **No Recommendations**: Pure analysis only - no style suggestions

## Installation

```bash
pip install facial-vision-agent
```

## Usage

```python
from facial_vision_agent import FacialVisionAgent

agent = FacialVisionAgent(openrouter_api_key="your_key")

# Comprehensive analysis
response = agent.process(AgentTask(
    type="analyze_image",
    payload={"image_path": "photo.jpg"}
))

# Specific analyses
response = agent.process(AgentTask(
    type="detect_facial_features", 
    payload={"image_path": "photo.jpg"}
))
```

## Output

Returns structured analysis data without style recommendations.
