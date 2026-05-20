#!/usr/bin/env python3
"""
Export blogs from Markdown to JSON format for v0/web integration
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict

BLOG_FOLDER = "blog"
OUTPUT_FILE = "blogs_export.json"

def parse_frontmatter_and_content(content: str) -> tuple:
    """Parse YAML frontmatter and markdown content"""
    # Remove markdown code fence if present
    if content.startswith("```markdown\n"):
        content = content[12:]  # Remove "```markdown\n"
    
    # Extract frontmatter
    frontmatter_pattern = r'^---\n(.*?)\n---\n(.*)$'
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    
    if not match:
        return {}, content
    
    frontmatter_text = match.group(1)
    body = match.group(2)
    
    # Parse YAML frontmatter
    metadata = {}
    for line in frontmatter_text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    
    return metadata, body

def extract_blogs() -> List[Dict]:
    """Extract all blog posts and their metadata"""
    blogs = []
    blog_path = Path(BLOG_FOLDER)
    
    # Get all markdown files except index.md
    md_files = sorted([f for f in blog_path.glob("*.md") if f.name != "index.md"])
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        metadata, body = parse_frontmatter_and_content(content)
        slug = md_file.stem
        
        blog_data = {
            "slug": slug,
            "filename": md_file.name,
            "title": metadata.get("title", ""),
            "description": metadata.get("meta_description", ""),
            "keyword": metadata.get("keyword", ""),
            "date": metadata.get("date", ""),
            "content": body.strip(),
            "excerpt": body.split('\n')[0][:200] if body else ""
        }
        
        blogs.append(blog_data)
    
    return blogs

def export_to_json(blogs: List[Dict]) -> None:
    """Export blogs to JSON file"""
    output_data = {
        "total": len(blogs),
        "blogs": blogs
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(blogs)} blogs to {OUTPUT_FILE}")

def create_csv_export(blogs: List[Dict]) -> None:
    """Create CSV for easy viewing in spreadsheets"""
    import csv
    
    csv_file = "blogs_export.csv"
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "keyword", "date", "description", "slug"])
        writer.writeheader()
        
        for blog in blogs:
            writer.writerow({
                "title": blog["title"],
                "keyword": blog["keyword"],
                "date": blog["date"],
                "description": blog["description"],
                "slug": blog["slug"]
            })
    
    print(f"✅ Exported {len(blogs)} blogs to {csv_file}")

if __name__ == "__main__":
    print("🚀 Extracting blogs...")
    
    try:
        blogs = extract_blogs()
        export_to_json(blogs)
        create_csv_export(blogs)
        
        print(f"\n📊 Summary:")
        print(f"   Total blogs: {len(blogs)}")
        print(f"   Files created:")
        print(f"   - blogs_export.json (pour v0)")
        print(f"   - blogs_export.csv (pour vue rapide)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
