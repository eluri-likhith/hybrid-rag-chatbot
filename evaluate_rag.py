

from rag_engine import answer_question

# Ground-truth relevant chunk IDs for each question.
# Replace these example IDs with IDs verified from your document.
ground_truth = {
    "What is the annual leave policy?": {0},
    "What is the sick leave policy?": {1, 2},
    "How can employees apply for leave?": {3, 5, 6},
}


def evaluate_retrieval(retrieved_docs, relevant_ids, k=3):
    top_k_results = retrieved_docs[:k]

    retrieved_ids = [
        doc.metadata.get("chunk_id")
        for doc in top_k_results
    ]

    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        raise ValueError("Ground-truth relevant IDs cannot be empty.")

    relevant_retrieved = set(retrieved_ids) & relevant_ids

    precision = (
        len(relevant_retrieved) / len(retrieved_ids)
        if retrieved_ids else 0.0
    )
    recall = len(relevant_retrieved) / len(relevant_ids)

    mrr = 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            mrr = 1 / rank
            break

    print("\nRetrieved Chunk IDs:", retrieved_ids)
    print("Ground Truth IDs:", sorted(relevant_ids))
    print("Relevant Retrieved IDs:", relevant_retrieved)
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}:    {recall:.4f}")
    print(f"MRR:           {mrr:.4f}")

    return precision, recall, mrr


if __name__ == "__main__":
    file_path = input("Enter your PDF/DOCX path: ").strip().strip('"')

    with open(file_path, "rb") as file:
        file_bytes = file.read()

    file_name = file_path.replace("\\", "/").split("/")[-1]

    all_metrics = []

    for question, relevant_ids in ground_truth.items():
        print(f"\n{'=' * 50}")
        print("Question:", question)

        answer, sources, reranked_results = answer_question(
            file_bytes=file_bytes,
            file_name=file_name,
            user_query=question,
        )

        print("\nGenerated Answer:")
        print(answer)

        precision, recall, mrr = evaluate_retrieval(
            retrieved_docs=reranked_results,
            relevant_ids=relevant_ids,
            k=3,
        )

        all_metrics.append((precision, recall, mrr))

    # Average metrics across all test questions
    if all_metrics:
        count = len(all_metrics)
        print("\n===== OVERALL RAG RETRIEVAL EVALUATION =====")
        print(f"Mean Precision@3: {sum(x[0] for x in all_metrics) / count:.4f}")
        print(f"Mean Recall@3:    {sum(x[1] for x in all_metrics) / count:.4f}")
        print(f"Mean MRR:         {sum(x[2] for x in all_metrics) / count:.4f}")
