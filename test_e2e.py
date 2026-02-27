import requests
import sys

BASE_URL = 'http://localhost:5001/api'
print("Starting API E2E Tests...\n")

def run_test(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
    except Exception as e:
        print(f"FAIL: {name}")
        print(f"   Error: {e}")
        sys.exit(1)

# Tokens
seeker_token = None
employer_token = None
resume_id = None
job_id = None

def t_register_seeker():
    global seeker_token
    res = requests.post(f"{BASE_URL}/auth/register", json={
        "role": "seeker",
        "email": "test_seeker_99@test.com",
        "password": "pass",
        "first_name": "Test",
        "last_name": "Seeker"
    })
    
    # If already exists or created
    if res.status_code == 400 and "already registered" in res.text:
        res = requests.post(f"{BASE_URL}/auth/login", json={
            "role": "seeker",
            "email": "test_seeker_99@test.com",
            "password": "pass"
        })
    assert res.status_code == 200, f"Failed seeker auth: {res.text}"
    seeker_token = res.json()['token']

def t_register_employer():
    global employer_token
    res = requests.post(f"{BASE_URL}/auth/register", json={
        "role": "employer",
        "email": "test_emp_99@test.com",
        "password": "pass",
        "first_name": "Test",
        "last_name": "Employer",
        "company": "Test Co"
    })
    
    if res.status_code == 400 and "already registered" in res.text:
        res = requests.post(f"{BASE_URL}/auth/login", json={
            "role": "employer",
            "email": "test_emp_99@test.com",
            "password": "pass"
        })
    assert res.status_code == 200, f"Failed employer auth: {res.text}"
    employer_token = res.json()['token']

def t_extract_resume():
    global resume_id
    res = requests.post(f"{BASE_URL}/resumes", 
        json={"text": "I am a skilled Python developer. I know Machine Learning, SQL, and Pandas. Location: Bangalore. Salary: 15-20 LPA", "name": "Python Dev Resume"},
        headers={"Authorization": f"Bearer {seeker_token}"}
    )
    assert res.status_code == 200, f"Failed to extract resume: {res.text}"
    data = res.json()
    assert 'python' in data['skills'], "Failed skill extraction"
    resume_id = data['id']

def t_match_jobs():
    res = requests.post(f"{BASE_URL}/match/jobs", 
        json={"resume_id": resume_id},
        headers={"Authorization": f"Bearer {seeker_token}"}
    )
    assert res.status_code == 200, f"Failed to match jobs: {res.text}"
    data = res.json()
    assert len(data) > 0, "Expected at least 1 job match"
    assert "scores" in data[0], "No score in response"

def t_post_job():
    global job_id
    res = requests.post(f"{BASE_URL}/jobs", 
        json={
            "title": "Machine Learning Engineer",
            "desc": "Join our ML team",
            "skills": "python, machine learning, sql, pandas",
            "exp": "2-5",
            "location": "bangalore",
            "salary": "15-25"
        },
        headers={"Authorization": f"Bearer {employer_token}"}
    )
    assert res.status_code == 200, f"Failed to post job: {res.text}"
    job_id = res.json()['id']

def t_match_candidates():
    res = requests.post(f"{BASE_URL}/match/candidates", 
        json={
            "skills": "python, machine learning, sql, pandas",
            "exp": "2-5",
            "location": "bangalore",
            "salary": "15-25"
        },
        headers={"Authorization": f"Bearer {employer_token}"}
    )
    assert res.status_code == 200, f"Failed to match candidates: {res.text}"
    data = res.json()
    assert len(data) > 0, "Expected at least 1 candidate match"

def t_metrics():
    res = requests.get(f"{BASE_URL}/metrics")
    assert res.status_code == 200, "Failed to get metrics"
    assert "Precision@5" in res.json()

run_test("Seeker Auth (Register/Login)", t_register_seeker)
run_test("Employer Auth (Register/Login)", t_register_employer)
run_test("Seeker: Extract Resume to DB", t_extract_resume)
run_test("Seeker: Match Jobs with AI Engine", t_match_jobs)
run_test("Employer: Post New Job to DB", t_post_job)
run_test("Employer: Find Top Candidates", t_match_candidates)
run_test("System: Fetch Model Metrics", t_metrics)

print("\nALL TESTS PASSED! Backend is 100% operational.")
