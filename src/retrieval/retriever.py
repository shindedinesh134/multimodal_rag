def retrieve(query, db):
    return db.similarity_search(query, k=3)