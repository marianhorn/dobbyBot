from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import nest_asyncio

# Allow nested event loops (useful for Jupyter or environments with active event loops)
nest_asyncio.apply()

# Load TinyLlama tokenizer and model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Define an asynchronous function to generate responses
async def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=200, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Define Telegram message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    bot_response = await generate_response(user_message)  # Use await to handle async response generation
    await update.message.reply_text(bot_response)

# Main function to start the bot
async def main():
    TOKEN = "8133748444:AAFQYTPCjhY07ADwasy4RfVYWYvndwaZkuk"  # Replace with your actual bot token
    application = Application.builder().token(TOKEN).build()

    # Add message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot with polling
    await application.run_polling()

# Script entry point
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If an event loop is already running, create a task for the main function
            asyncio.create_task(main())
        else:
            # Run the main function in a new event loop
            asyncio.run(main())
    except RuntimeError:
        # Fallback if no loop exists
        asyncio.run(main())
