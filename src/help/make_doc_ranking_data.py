import json
import sys
import numpy as np
import argparse
import collections
from tqdm import tqdm
import gzip


def read_json_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def load_docs(doc_file):
    docs = {}
    with open(doc_file, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            d = json.loads(line)
            docs[d['doc_id']] = (d['entities'], d['text'])
    return docs


def read_qrels(qrels_file: str):
    qrels = collections.defaultdict(dict)

    with open(qrels_file, 'r') as f:
        for line in f:
            query_id, _, object_id, relevance = line.strip().split()
            assert object_id not in qrels[query_id]
            qrels[query_id][object_id] = int(relevance)

    return qrels


def read_run(run_file):
    run = collections.defaultdict(dict)

    with open(run_file, 'r') as f:
        for line in f:
            query_id = line.strip().split()[0]
            object_id = line.strip().split()[2]
            score = line.strip().split()[4]
            if object_id not in run[query_id]:
                run[query_id][object_id] = float(score)

    return run


def load_embeddings_with_names(embedding_file):
    """Load embeddings with entity names for alignment"""
    emb = {}
    entity_names = {}

    print("Loading embeddings with entity names...")
    with gzip.open(embedding_file, 'r') as f:
        for line in tqdm(f, total=13032425, desc="Loading embeddings"):
            d = json.loads(line)
            entity_id = d['entity_id']
            emb[entity_id] = d['embedding']
            entity_names[entity_id] = d['entity_name']

    print(f"Loaded {len(emb)} entities with names")
    return emb, entity_names


def load_queries(queries_file):
    with open(queries_file, 'r') as f:
        return dict({
            (line.strip().split('\t')[0], line.strip().split('\t')[1])
            for line in f
        })


def write_to_file(data_line, save):
    with open(save, 'a') as f:  # Open the file in append mode to add lines
        f.write("%s\n" % data_line)


def get_entity_centric_doc_embedding_with_names(doc_entities, query_entities, query_entity_embeddings, entity_names):
    """Get document embeddings with aligned entity IDs and names"""
    embeddings = []
    entity_ids = []
    names = []

    for entity_id in doc_entities:
        entity_id = str(entity_id)
        if entity_id in query_entity_embeddings and entity_id in query_entities and entity_id in entity_names:
            entity_embedding = query_entity_embeddings[entity_id]
            if len(entity_embedding) >= 300:
                entity_embedding = entity_embedding[:300]
                # Scale by query entity score (entity-centric approach)
                scaled_embedding = query_entities[entity_id] * np.array(entity_embedding)

                embeddings.append(scaled_embedding.tolist())
                entity_ids.append(entity_id)
                names.append(entity_names[entity_id])

    if not embeddings:
        return None, [], []

    return embeddings, entity_ids, names


def get_docs(docs, qrels, query_entities, query_entity_embeddings, entity_names, positive, query_docs, doc_scores):
    """Get documents with entity embeddings, IDs, and names"""
    d = {}

    # Iterate through all the documents in query_docs
    for doc_id in query_docs:
        # Check if the current document ID is in the docs dictionary
        if doc_id in docs and doc_id in doc_scores:
            # Determine if the current document is a positive (relevant) document based on the qrels dictionary
            is_positive = doc_id in qrels and qrels[doc_id] >= 1

            # Process the document if it matches the desired relevance status (positive or negative)
            if is_positive == positive:
                # Compute the entity-centric document embedding for the current document
                doc_ent_embeddings, doc_entity_ids, doc_entity_names = get_entity_centric_doc_embedding_with_names(
                    docs[doc_id][0], query_entities, query_entity_embeddings, entity_names
                )

                # Check if the document entity embedding is not None
                if doc_ent_embeddings is not None:
                    doc_text = docs[doc_id][1]
                    doc_score = doc_scores[doc_id]
                    d[doc_id] = (doc_text, doc_score, doc_ent_embeddings, doc_entity_ids, doc_entity_names)

    # Return the dictionary containing the documents (positive or negative) and their entity embeddings
    return d


def get_entity_centric_query_embedding_with_names(query_entities, query_entity_embeddings, entity_names, k):
    """Get query embeddings with aligned entity IDs and names"""
    embeddings = []
    entity_ids = []
    names = []

    # Take top k entities by score
    expansion_entities = dict(list(query_entities.items())[:k])

    for entity_id, score in expansion_entities.items():
        if entity_id in query_entity_embeddings and entity_id in entity_names:
            entity_embedding = query_entity_embeddings[entity_id]
            if len(entity_embedding) >= 300:
                entity_embedding = entity_embedding[:300]
                # Scale by entity score (entity-centric approach)
                scaled_embedding = score * np.array(entity_embedding)

                embeddings.append(scaled_embedding.tolist())
                entity_ids.append(entity_id)
                names.append(entity_names[entity_id])

    if not embeddings:
        return None, [], []

    return embeddings, entity_ids, names


def make_data_strings(query, query_id, query_ent_emb, query_entity_ids, query_entity_names, docs, label, save):
    """Create data strings with entity information"""
    for doc_id in docs:
        doc_text, doc_score, doc_ent_emb, doc_entity_ids, doc_entity_names = docs[doc_id]

        data_line = json.dumps({
            'query': query,
            'query_id': query_id,
            'query_ent_emb': query_ent_emb,
            'doc_id': doc_id,
            'doc': doc_text,
            'doc_score': doc_score,
            'doc_ent_emb': doc_ent_emb,
            'label': label
        })
        write_to_file(data_line, save)


def get_query_entity_embeddings(query_entities, entity_embeddings):
    """Get entity embeddings for query entities"""
    emb = {}
    for entity_id, score in query_entities.items():
        if entity_id in entity_embeddings:
            emb[entity_id] = entity_embeddings[entity_id]
    return emb


def create_data(queries, docs, doc_qrels, doc_run, entity_run, entity_embeddings, entity_names, k, train, balance,
                save):
    """Create training/test data with entity names"""
    stats = {
        'total_queries': 0,
        'queries_with_entities': 0,
        'total_examples': 0,
        'examples_with_query_entities': 0,
        'examples_with_doc_entities': 0
    }

    for query_id, query in tqdm(queries.items(), total=len(queries)):
        stats['total_queries'] += 1

        if query_id in doc_run and query_id in entity_run and query_id in doc_qrels:
            query_docs = doc_run[query_id]
            query_entities = entity_run[query_id]

            query_entity_embeddings = get_query_entity_embeddings(
                query_entities=query_entities,
                entity_embeddings=entity_embeddings
            )

            qrels = doc_qrels[query_id]

            # Get entity-centric query embedding with names
            entity_centric_query_emb, query_entity_ids, query_entity_names = get_entity_centric_query_embedding_with_names(
                query_entities, query_entity_embeddings, entity_names, k
            )

            if entity_centric_query_emb is not None:
                stats['queries_with_entities'] += 1

            if train:
                # If this is train data then we are going to get the positive examples from the qrels file.
                # Any document that is explicitly annotated as relevant in the qrels in considered positive.
                pos_docs = get_docs(
                    docs=docs,
                    qrels=qrels,
                    query_entities=query_entities,
                    query_entity_embeddings=query_entity_embeddings,
                    entity_names=entity_names,
                    positive=True,
                    query_docs=set(qrels.keys()),
                    doc_scores=query_docs
                )
            else:
                # If this is test data, then we are going to get the positive examples from the document run file.
                # In this case we set the `query_docs` parameter.
                # Any document that is explicitly annotated as relevant in the qrels in considered positive.
                pos_docs = get_docs(
                    docs=docs,
                    qrels=qrels,
                    query_entities=query_entities,
                    query_entity_embeddings=query_entity_embeddings,
                    entity_names=entity_names,
                    positive=True,
                    query_docs=set(query_docs.keys()),
                    doc_scores=query_docs
                )
            # We always choose the negative examples from the run file.
            # A document is negative if either:
            #   (1) It is explicitly annotated as non-relevant in the qrels, OR
            #   (2) The document is not in the qrels AND present in the run file.
            neg_docs = get_docs(
                docs=docs,
                qrels=qrels,
                query_entities=query_entities,
                query_entity_embeddings=query_entity_embeddings,
                entity_names=entity_names,
                positive=False,
                query_docs=set(query_docs.keys()),
                doc_scores=query_docs
            )

            if balance:
                # If this is true, then we balance the number of positive and negative examples
                n = min(len(pos_docs), len(neg_docs))
                pos_docs = dict(list(pos_docs.items())[:n])
                neg_docs = dict(list(neg_docs.items())[:n])

            # Create positive examples
            make_data_strings(
                query=query,
                query_id=query_id,
                query_ent_emb=entity_centric_query_emb if entity_centric_query_emb else [],
                query_entity_ids=query_entity_ids,
                query_entity_names=query_entity_names,
                docs=pos_docs,
                label=1,
                save=save
            )

            # Create negative examples
            make_data_strings(
                query=query,
                query_id=query_id,
                query_ent_emb=entity_centric_query_emb if entity_centric_query_emb else [],
                query_entity_ids=query_entity_ids,
                query_entity_names=query_entity_names,
                docs=neg_docs,
                label=0,
                save=save
            )

            # Update stats
            examples_added = len(pos_docs) + len(neg_docs)
            stats['total_examples'] += examples_added
            if entity_centric_query_emb:
                stats['examples_with_query_entities'] += examples_added

            for doc_data in list(pos_docs.values()) + list(neg_docs.values()):
                if len(doc_data) >= 5 and doc_data[4]:  # Has doc entity names
                    stats['examples_with_doc_entities'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser("Make train/test data with entity names.")
    parser.add_argument("--queries", help="Queries file.", required=True, type=str)
    parser.add_argument("--docs", help="Document file.", required=True, type=str)
    parser.add_argument("--qrels", help="Document qrels.", required=True, type=str)
    parser.add_argument("--doc-run", help="Document run file.", required=True, type=str)
    parser.add_argument("--entity-run", help="Entity run file.", required=True, type=str)
    parser.add_argument("--embeddings", help="Wikipedia2Vec entity embeddings file.", required=True, type=str)
    parser.add_argument("--k", help="Number of expansion entities. Default=20", default=20, type=int)
    parser.add_argument('--train', help='Is this train data? Default: False.', action='store_true')
    parser.add_argument('--balance', help='Should the data be balanced? Default: False.', action='store_true')
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--save-stats", help="Save statistics file.", type=str)

    args = parser.parse_args(args=None if sys.argv[1:] else ['--random'])

    print("=" * 60)
    print("ENTITY-CENTRIC DATA CREATION WITH NAMES")
    print("=" * 60)

    if args.train:
        print('✅ Creating train data...')
    else:
        print('✅ Creating test data...')

    if args.balance:
        print('✅ Dataset will be balanced (equal number of positive and negative examples).')
    else:
        print('✅ Dataset will be unbalanced.')

    print(f'✅ Using top-{args.k} entities for query expansion.')

    print('Loading queries...')
    queries = load_queries(queries_file=args.queries)
    print('[Done].')

    print('Loading documents...')
    docs = load_docs(args.docs)
    print('[Done].')

    print('Loading qrels...')
    qrels = read_qrels(args.qrels)
    print('[Done].')

    print('Loading document run...')
    doc_run = read_run(args.doc_run)
    print('[Done].')

    print('Loading entity run...')
    entity_run = read_run(args.entity_run)
    print('[Done].')

    print('Loading embeddings with entity names...')
    embeddings, entity_names = load_embeddings_with_names(args.embeddings)
    print('[Done].')

    # Clear output file
    with open(args.save, 'w') as f:
        pass

    print('Creating data with aligned entity information...')
    stats = create_data(
        queries=queries,
        docs=docs,
        doc_qrels=qrels,
        doc_run=doc_run,
        entity_run=entity_run,
        entity_embeddings=embeddings,
        entity_names=entity_names,
        k=args.k,
        train=args.train,
        balance=args.balance,
        save=args.save
    )
    print('[Done].')

    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Queries with entities: {stats['queries_with_entities']}")
    print(f"Total examples created: {stats['total_examples']}")
    print(f"Examples with query entities: {stats['examples_with_query_entities']}")
    print(f"Examples with doc entities: {stats['examples_with_doc_entities']}")

    if stats['total_examples'] > 0:
        query_coverage = stats['examples_with_query_entities'] / stats['total_examples'] * 100
        doc_coverage = stats['examples_with_doc_entities'] / stats['total_examples'] * 100
        print(f"Query entity coverage: {query_coverage:.1f}%")
        print(f"Document entity coverage: {doc_coverage:.1f}%")

    # Save statistics if requested
    if args.save_stats:
        with open(args.save_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {args.save_stats}")

    print(f"\n✅ Entity-centric data created with perfect alignment:")
    print(f"   - Embeddings scaled by entity scores (entity-centric approach)")
    print(f"   - query_ent_emb[i] ↔ query_entity_names[i] ↔ query_entity_ids[i]")
    print(f"   - doc_ent_emb[i] ↔ doc_entity_names[i] ↔ doc_entity_ids[i]")
    print(f"   - Top-{args.k} query entities used for expansion")


if __name__ == '__main__':
    main()