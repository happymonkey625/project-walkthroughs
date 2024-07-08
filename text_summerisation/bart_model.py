from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def chunk_text(text, tokenizer, max_chunk_length):
    inputs = tokenizer(text, return_tensors='pt', max_length=max_chunk_length, truncation=True)
    input_ids = inputs['input_ids'][0]

    # Split the token IDs into chunks within the max chunk length
    chunks = []
    for i in range(0, len(input_ids), max_chunk_length - 2):  # subtract 2 to avoid overflow with special tokens
        chunks.append(input_ids[i:i + max_chunk_length - 2])
    
    return chunks

def decode_chunks(chunks, tokenizer):
    texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return texts

def summarize_meeting_transcription(transcription, max_chunk_length=1024, max_length=150, min_length=50):
    # Load the model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Create the summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    # Split the transcription into chunks
    token_chunks = chunk_text(transcription, tokenizer, max_chunk_length)
    text_chunks = decode_chunks(token_chunks, tokenizer)
    
    # Summarize each chunk
    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Combine the summaries into a final summary
    final_summary = ' '.join(summaries)
    return final_summary

# Read the meeting transcription from the file
with open('/Users/leonachen/Downloads/combined_meeting_transcription.txt', 'r') as file:
    transcription = file.read()

# Get the summary
summary = summarize_meeting_transcription(transcription)

# Output the summary
print("Meeting Summary:\n")
print(summary)