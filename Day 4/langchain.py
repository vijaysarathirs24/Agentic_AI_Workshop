# -*- coding: utf-8 -*-
"""LAngchain.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H-WvmQ9dLiVjsB6jQpqCZZ8E-JrslhgD
"""

# Install required packages
!pip install -q google-generativeai tavily-python

# Imports
import os
import google.generativeai as genai
from tavily import TavilyClient
from getpass import getpass
from IPython.display import Markdown
from google.colab import files

# API keys (keep these secret!)
GOOGLE_API_KEY = getpass("Google Gemini API Key: ")
TAVILY_API_KEY = getpass("Tavily API Key: ")

# Set up the services
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")
tavily = TavilyClient(api_key=TAVILY_API_KEY)


class ResearchBot:
    def __init__(self, topic):
        self.topic = topic
        self.questions = []
        self.results = []

    def ask_gemini(self):
        print("Generating research questions...")
        prompt = f"List 5 to 6 insightful and varied research questions about: {self.topic}"
        response = gemini.generate_content(prompt)
        lines = response.text.strip().splitlines()

        self.questions = [
            line.lstrip("•-0123456789. ").strip()
            for line in lines if line.strip()
        ]

        print("\nQuestions:")
        for i, q in enumerate(self.questions, 1):
            print(f"{i}. {q}")

    def do_web_search(self):
        print("\nRunning searches...")
        for i, question in enumerate(self.questions, 1):
            query = question[:400]  # API limit
            print(f"→ [{i}] {query}")
            search_data = tavily.search(query=query, search_depth="advanced")
            self.results.append({
                "question": question,
                "entries": [
                    {
                        "title": item["title"],
                        "content": item["content"]
                    } for item in search_data.get("results", [])
                ]
            })

    def build_markdown_report(self):
        md = f"# Research Report: {self.topic}\n\n"
        md += "## Overview\n"
        md += f"This report presents findings based on web-sourced answers to questions about **{self.topic}**.\n\n"

        for item in self.results:
            md += f"### {item['question']}\n"
            for res in item['entries'][:3]:
                md += f"**{res['title']}**\n\n{res['content']}\n\n"

        md += "---\n### Final Notes\nThis research was compiled using Gemini and Tavily.\n"
        return md


# Run the flow
topic_input = input("Enter a topic to explore: ").strip()

if not topic_input:
    print("No topic entered. Try again.")
else:
    bot = ResearchBot(topic_input)
    bot.ask_gemini()
    bot.do_web_search()
    report_text = bot.build_markdown_report()

    output_file = "report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved report to: {output_file}")
    display(Markdown(report_text))
    files.download(output_file)