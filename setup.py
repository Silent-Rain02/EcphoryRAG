from setuptools import setup, find_packages
import re

with open("ecphoryrag/__init__.py", "r") as f:
    content = f.read()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ecphoryrag",
    version=version,
    author="EcphoryRAG Team",
    author_email="your.email@example.com",
    description="A neurocognitive-inspired RAG system using entity-based memory traces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/EcphoryRAG",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "ollama>=0.1.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.25.0",
        "requests>=2.31.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecphoryrag-demo=ecphoryrag.examples.run_demo:main",
            "ecphoryrag-api=ecphoryrag.api.main:app",
        ],
    },
) 