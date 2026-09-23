#variables & data types 
student_name = "alex"
target_role = "AI Integration Engineer"
tech_stack = ["python", "fastapi", "gemini", "langchain"]

print(f"--- {student_name}'s AI learning plan ---")
for index, tech in enumerate(tech_stack, 1):
    # Define a dictionary for hours dedicated to each skill
    hours_dedicated = {
        "python": 20,
        "fastapi": 15,
        "gemini": 10,
        "langchain": 12
    }
    print(f"{index}.{tech} - {hours_dedicated.get(tech, 0)} hours dedicated")
    
    #calculate total learning hours 
    total_hours = sum(hours_dedicated.values())

    #check commitment level
    if total_hours >=50:
        print("\ngreat! solid commitment for AI integration engineer role.")
    else:
        print("\nIncrease practice hours.")
