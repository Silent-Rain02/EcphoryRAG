# ollama_clients.py

import asyncio
import time
import requests
from ollama import AsyncClient, ResponseError, embed
from typing import List, Optional, Dict, Any, Tuple

# Ensure ollama library is installed: pip install ollama

# Global counters for token usage
embedding_token_count = 0
completion_token_count = 0

def reset_token_counters():
    """Reset all token usage counters to zero."""
    global embedding_token_count, completion_token_count
    embedding_token_count = 0
    completion_token_count = 0

def get_token_usage():
    """Get the current token usage counts."""
    return {
        "embedding_tokens": embedding_token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": embedding_token_count + completion_token_count
    }

def get_ollama_embedding(text: str | List[str], model_name: str = "bge-m3", max_retries: int = 3, retry_delay: float = 1.0) -> List[float] | List[List[float]]:
    """
    Generates an embedding for the given text using the specified Ollama embedding model.
    Includes retry mechanism to handle temporary failures.

    Args:
        text: The input text to embed (can be a single string or a list of strings).
        model_name: The name of the Ollama embedding model to use (default: "bge-m3").
        max_retries: Maximum number of retry attempts if embedding fails.
        retry_delay: Delay in seconds between retry attempts.

    Returns:
        For single text input: A list of floats representing the embedding.
        For batch input: A list of lists of floats representing the embeddings.
        Returns an empty list if an error occurs.
    """
    global embedding_token_count
    
    # Handle batch input
    if isinstance(text, list):
        if not text:
            return []
        results = []
        for t in text:
            if not t or not t.strip():
                results.append([])
                continue
            # Estimate token count (rough estimate: 4 chars ≈ 1 token)
            estimated_tokens = len(t) // 4
            embedding_token_count += estimated_tokens
            
            # Truncate text if it's too long
            max_length = 2048
            if len(t) > max_length:
                truncated_text = t[:max_length]
                print(f"Warning: Text truncated from {len(t)} to {max_length} characters for embedding.")
            else:
                truncated_text = t
            
            # Get embedding for this text
            embedding = _get_single_embedding(truncated_text, model_name, max_retries, retry_delay)
            results.append(embedding)
        return results
    
    # Handle single text input
    if not text or not text.strip():
        print("Warning: Empty text provided for embedding. Returning empty list.")
        return []
    
    # Estimate token count (rough estimate: 4 chars ≈ 1 token)
    estimated_tokens = len(text) // 4
    embedding_token_count += estimated_tokens
    
    # Truncate text if it's too long
    max_length = 2048
    if len(text) > max_length:
        truncated_text = text[:max_length]
        print(f"Warning: Text truncated from {len(text)} to {max_length} characters for embedding.")
    else:
        truncated_text = text
    
    return _get_single_embedding(truncated_text, model_name, max_retries, retry_delay)

def _get_single_embedding(text: str, model_name: str, max_retries: int, retry_delay: float) -> List[float]:
    """
    Internal helper function to get embedding for a single text.
    """
    for attempt in range(max_retries):
        try:
            # 调用Ollama API获取嵌入向量
            response = embed(model=model_name, input=text)
            
            # Ollama 0.4.8+版本返回EmbedResponse对象，有embeddings属性
            if hasattr(response, 'embeddings') and response.embeddings:
                # 对于单个输入，embeddings是一个列表的列表，取第一个元素
                if isinstance(response.embeddings, list) and len(response.embeddings) > 0:
                    return response.embeddings[0]
                return []
            
            # 兼容字典格式响应（旧版API）
            if isinstance(response, dict):
                # 尝试不同可能的键
                if 'embeddings' in response and response['embeddings']:
                    if isinstance(response['embeddings'], list) and len(response['embeddings']) > 0:
                        return response['embeddings'][0]
                    return response['embeddings']
                
                if 'embedding' in response and response['embedding']:
                    return response['embedding']
            
            # 响应格式不符合预期
            print(f"Warning: Unexpected embed response format on attempt {attempt+1}/{max_retries}.")
            if hasattr(response, '__dict__'):
                print(f"Response attributes: {dir(response)[:10]}...")
            else:
                print(f"Response type: {type(response)}")
                if isinstance(response, dict):
                    print(f"Response keys: {list(response.keys())}")
            
        except ResponseError as e:
            print(f"Ollama API error on attempt {attempt+1}/{max_retries} while getting embedding for model '{model_name}': {getattr(e, 'status_code', 'N/A')} - {e.error}")
        except requests.exceptions.ConnectionError as e:
            print(f"Ollama connection error on attempt {attempt+1}/{max_retries} while getting embedding for model '{model_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred on attempt {attempt+1}/{max_retries} while getting embedding for model '{model_name}': {e}")
        
        # If we've used all retries, return empty list
        if attempt >= max_retries - 1:
            print(f"Failed to get embedding after {max_retries} attempts. Returning empty list.")
            return []
        
        # Wait before retrying
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    # This should never be reached, but just in case
    return []

async def _async_get_ollama_completion(prompt: str, model_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Asynchronously gets a completion from the specified Ollama LLM.
    Helper function for get_ollama_completion.
    
    Returns:
        Tuple containing (completion_text, token_usage_info)
    """
    if not prompt or not prompt.strip():
        print("Warning: Empty prompt provided for completion. Returning empty string.")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    messages = [{'role': 'user', 'content': prompt}]
    try:
        # Assumes AsyncClient connects to default Ollama host/port (http://localhost:11434)
        # User can configure OLLAMA_HOST environment variable if needed.
        client = AsyncClient()
        response = await client.chat(model=model_name, messages=messages)
        
        # Safely access nested keys
        message_content = response.get('message', {}).get('content', '')
        
        # Extract token usage if available
        token_info = {
            "prompt_tokens": response.get('prompt_eval_count', 0),
            "completion_tokens": response.get('eval_count', 0),
            "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
        }
        
        return message_content, token_info
    except ResponseError as e:
        print(f"Ollama API error while getting completion from model '{model_name}': {getattr(e, 'status_code', 'N/A')} - {e.error}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    except requests.exceptions.ConnectionError as e:
        print(f"Ollama connection error while getting completion from model '{model_name}': {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    except Exception as e:
        print(f"An unexpected error occurred while getting completion from model '{model_name}': {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def get_ollama_completion(prompt: str, model_name: str = "phi3", max_retries: int = 2, retry_delay: float = 1.0) -> Tuple[str, Dict[str, int]]:
    """
    Gets a completion for the given prompt from the specified Ollama LLM.
    This is a synchronous wrapper around an asynchronous Ollama call.

    Args:
        prompt: The input prompt for the LLM.
        model_name: The name of the Ollama LLM to use (default: "phi3").
        max_retries: Maximum number of retry attempts if completion fails.
        retry_delay: Delay in seconds between retry attempts.

    Returns:
        Tuple containing (completion_text, token_usage_info)
    """
    global completion_token_count
    
    token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for attempt in range(max_retries):
        try:
            # This runs the async function and waits for its result.
            completion, usage = asyncio.run(_async_get_ollama_completion(prompt, model_name))
            
            # Update token usage
            if usage:
                token_info = usage
                completion_token_count += usage.get("total_tokens", 0)
                
            if completion:  # Check if we got a valid completion
                return completion, token_info
            
            # If we get here, completion is empty but no exception was raised
            print(f"Warning: Empty completion returned on attempt {attempt+1}/{max_retries}.")
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                print(
                    f"Error: 'get_ollama_completion' (which uses asyncio.run()) was called from a running asyncio event loop. "
                    f"If you are in an async context, consider awaiting '_async_get_ollama_completion(\\'{prompt}\\'*, \\'{model_name}\\')' directly. Details: {e}"
                )
                return "", token_info
            else:
                print(f"A runtime error occurred in get_ollama_completion on attempt {attempt+1}/{max_retries}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in get_ollama_completion setup/execution on attempt {attempt+1}/{max_retries}: {e}")
    
            # If we've used all retries, return empty string
            if attempt >= max_retries - 1:
                print(f"Failed to get completion after {max_retries} attempts. Returning empty string.")
                return "", token_info
            
            # Wait before retrying
            print(f"Retrying completion in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    # This should never be reached, but just in case
    return "", token_info

if __name__ == '__main__':
    # This section demonstrates usage and requires an Ollama server to be running.
    # You might need to pull the models first:
    # ollama pull bge-m3
    # ollama pull phi3

    print("--- Testing Ollama Clients ---")

    # Reset token counters
    reset_token_counters()

    # 1. Test embedding function
    print("\n[Embedding Test]")
    example_text_for_embedding = "The quick brown fox jumps over the lazy dog."
    print(f"Requesting embedding for: '{example_text_for_embedding}' with model 'bge-m3'")
    embedding_vector = get_ollama_embedding(example_text_for_embedding, model_name="bge-m3")

    if embedding_vector:
        print(f"Successfully retrieved embedding. Vector dimensions: {len(embedding_vector)}")
        print(f"First 5 dimensions: {embedding_vector[:5]}...")
    else:
        print("Failed to get embedding. Check if Ollama server is running and 'bge-m3' model is available (ollama pull bge-m3).")

    # Test embedding with a non-existent model
    print("\n[Embedding Test - Non-existent Model]")
    non_existent_embedding = get_ollama_embedding("test", model_name="non-existent-model-for-embedding")
    if not non_existent_embedding:
        print("Correctly handled non-existent model for embedding (returned empty list).")
    else:
        print("Error: Expected empty list for non-existent embedding model.")
    
    # Test embedding with empty text
    print("\n[Embedding Test - Empty Text]")
    empty_text_embedding = get_ollama_embedding("")
    if not empty_text_embedding:
        print("Correctly handled empty text for embedding (returned empty list).")
    else:
        print("Error: Expected empty list for empty text embedding.")


    # 2. Test completion function
    print("\n[Completion Test]")
    example_prompt_for_completion = "Explain the concept of 'entity' in the context of knowledge graphs."
    print(f"Requesting completion for prompt: '{example_prompt_for_completion}' with model 'phi3'")
    completion_text, token_usage = get_ollama_completion(example_prompt_for_completion, model_name="phi3")

    if completion_text:
        print("Successfully retrieved completion:")
        print(f"Prompt: '{example_prompt_for_completion}'")
        print(f"Completion: '{completion_text}'")
        print(f"Token usage: {token_usage}")
    else:
        print("Failed to get completion. Check if Ollama server is running and 'phi3' model is available (ollama pull phi3).")

    # Test completion with a non-existent model
    print("\n[Completion Test - Non-existent Model]")
    non_existent_completion = get_ollama_completion("test", model_name="non-existent-model-for-completion")
    if not non_existent_completion:
        print("Correctly handled non-existent model for completion (returned empty string).")
    else:
        print("Error: Expected empty string for non-existent completion model.")

    # Test completion with empty prompt
    print("\n[Completion Test - Empty Prompt]")
    empty_prompt_completion = get_ollama_completion("")
    if not empty_prompt_completion:
        print("Correctly handled empty prompt for completion (returned empty string).")
    else:
        print("Error: Expected empty string for empty prompt completion.")
        
    # Print token usage
    print("\n[Token Usage Summary]")
    usage = get_token_usage()
    print(f"Embedding tokens: {usage['embedding_tokens']}")
    print(f"Completion tokens: {usage['completion_tokens']}")
    print(f"Total tokens: {usage['total_tokens']}")
        
    print("\n--- Ollama Client Tests Complete ---")
    print("Note: For successful tests, ensure your Ollama server is running and accessible,")
    print("and the models 'bge-m3' and 'phi3' are downloaded (e.g., 'ollama pull bge-m3').") 