
from benchmarks.evaluate_thresholds import get_topic


def test_classifies_docker_questions():
    assert get_topic("What does Docker do?") == "docker"
    assert get_topic("How do containers work?") == "docker"


def test_classifies_kubernetes_questions():
    assert get_topic("Explain Kubernetes.") == "kubernetes"
    assert get_topic("What is K8s?") == "kubernetes"


def test_classifies_terraform_questions():
    assert get_topic("How does Terraform work?") == "terraform"


def test_classifies_aws_questions():
    assert get_topic("Explain cloud computing.") == "aws"
    assert get_topic("What is AWS?") == "aws"


def test_classifies_python_questions():
    assert get_topic("What is Python used for?") == "python"


def test_classifies_unrelated_questions():
    assert get_topic("How does DNS work?") == "other"
    assert get_topic("Explain version control.") == "other"
