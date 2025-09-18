#include <iostream>
#include <pthread.h>
#include <omp.h>

using std::cout;

bool debug = false;

#define NUM_THREADS (8)
#define BIG_PRIME (1000000007)

// (key, value) pair for hash table
struct KeyValue {
  bool empty;
	uint32_t key;
  uint32_t value;
	pthread_mutex_t lock;

	// Constructor
	KeyValue() : empty(true), lock(PTHREAD_MUTEX_INITIALIZER) {}
	KeyValue(uint32_t _key, uint32_t _value) : empty(false), key(_key), value(_value), lock(PTHREAD_MUTEX_INITIALIZER) {}
};

// (key, value) pair for users
struct UserKeyValue {
	uint32_t key;
	uint32_t value;
};

class HashTable {
private:
	uint64_t MAX_ELEMENTS; // maximum number of elements in the hash table 
	KeyValue *table;

	void resize(){
			#pragma omp parallel for
			for(uint64_t i = 0; i < MAX_ELEMENTS; i++) pthread_mutex_lock(&table[i].lock);
			
			MAX_ELEMENTS <<= 1;

			KeyValue *new_table = (KeyValue*)malloc(MAX_ELEMENTS * sizeof(KeyValue));
		
			#pragma omp parallel for
			for(uint64_t i = 0; i < MAX_ELEMENTS; i++) new_table[i] = KeyValue();
			
			#pragma omp parallel for
			for(uint64_t i = 0; i < (MAX_ELEMENTS >> 1); i++) insert_into_new_table(table[i].key, table[i].value, new_table);

			#pragma omp parallel for
			for(uint64_t i = 0; i < (MAX_ELEMENTS >> 1); i++) pthread_mutex_unlock(&table[i].lock);

			table = new_table;
	}
	
	// primary hash
	uint64_t h(uint32_t key){
		return (key * key) % MAX_ELEMENTS;
	}

	// secondary hash
	uint64_t g(uint32_t key){
		return (key * key) % BIG_PRIME;
	}

	bool insert_into_new_table(uint32_t key, uint32_t value, KeyValue *new_table){
			if(debug) cout << "insert into new table called for key = " << key << ", value = " << value << "\n";
			if(search_in_new_table(key, new_table)) return false;
			uint64_t base = h(key), offset = g(key);
			uint64_t curr = base;
			do{
					if(new_table[curr].empty){
							new_table[curr].key = key;
							new_table[curr].value = value;
							return true;
					}
					curr = (curr + offset) % MAX_ELEMENTS;
			} while (curr != base);
			cout << "ERROR: not able to insert into new table\n";
			return false;
	}

	bool search_in_new_table(uint32_t key, KeyValue *new_table){
			if(debug) cout << "search in new table called for key = " << key << "\n";
			uint64_t base = h(key), offset = g(key);
			uint64_t curr = base;
			do{
					if(!new_table[curr].empty && new_table[curr].key == key) return true;
					curr = (curr + offset) % MAX_ELEMENTS;
			} while (curr != base);
			return false;
	}

	bool insert(uint32_t key, uint32_t value){
			if(debug) cout << "insert called for key = " << key << ", value = " << value << "\n";
			if(search(key)) return false;

			uint64_t base = h(key), offset = g(key);
			uint64_t curr = base;
			do{
					pthread_mutex_lock(&table[curr].lock);
					if(table[curr].empty){
							table[curr].key = key;
							table[curr].value = value;
							pthread_mutex_unlock(&table[curr].lock);
							return true;
					}
					pthread_mutex_unlock(&table[curr].lock);
					curr = (curr + offset) % MAX_ELEMENTS;
			} while (curr != base);
			resize();
			return insert(key, value);
	}

	bool del(uint32_t key){
			if(debug) cout << "del called for key = " << key << "\n";
			if(!search(key)) return false;

			uint64_t base = h(key), offset = g(key);
			uint64_t curr = base;
			do{
					pthread_mutex_lock(&table[curr].lock);
					if(!table[curr].empty && table[curr].key == key){
							table[curr].empty = true;
							pthread_mutex_unlock(&table[curr].lock);
							return true;
					}
					pthread_mutex_unlock(&table[curr].lock);
					curr = (curr + offset) % MAX_ELEMENTS;
			} while (curr != base);
			
			// after searching and before deleting, some other thread might delete this key
			return false;
	}

	bool search(uint32_t key){
			if(debug) cout << "search called for key = " << key << "\n";
			uint64_t base = h(key), offset = g(key);
			uint64_t curr = base;
			do{
					pthread_mutex_lock(&table[curr].lock);
					if(!table[curr].empty && table[curr].key == key){
							pthread_mutex_unlock(&table[curr].lock);
							return true;
					}
					pthread_mutex_unlock(&table[curr].lock);
					curr = (curr + offset) % MAX_ELEMENTS;
			} while (curr != base);
			return false;
	}

public:
		HashTable(){
			omp_set_num_threads(NUM_THREADS);
			MAX_ELEMENTS = 1;
			table = (KeyValue*)malloc(MAX_ELEMENTS * sizeof(KeyValue));
			table[0] = KeyValue();
		}

		void batch_insert(uint64_t ADD, UserKeyValue* kv_pairs, bool *result){
			if(debug) cout << "batch_insert called\n";
			#pragma omp parallel for
			for(uint64_t i = 0; i < ADD; i++) result[i] = insert(kv_pairs[i].key, kv_pairs[i].value);
		}

		void batch_delete(uint64_t REM, uint32_t *key_list, bool *result){
			if(debug) cout << "batch_delete called\n";
			#pragma omp parallel for
			for(uint64_t i = 0; i < REM; i++) result[i] = del(key_list[i]);
		}

		void batch_search(uint64_t FIND, uint32_t *key_list, uint32_t *result){
			if(debug) cout << "batch_search called\n";
			#pragma omp parallel for
			for(uint64_t i = 0; i < FIND; i++) result[i] = search(key_list[i]);
		}
};
