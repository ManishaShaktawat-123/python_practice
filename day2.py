from typing import Dict, Any

def analyze_student_performance(scores: Dict[str, int]) -> Dict[str, Any]:
    total_score = sum(scores.values())
    average_score = total_score / len(scores)
    status = "Passed" if average_score >= 50 else "Needs Improvement"
    return {
        "total": total_score,
        "average": round(average_score, 2),
        "status": status
    }

student_scores = {"algorithm": 85, "graphics": 78, "os": 92}
result = analyze_student_performance(student_scores)
print("Performance Summary:", result)