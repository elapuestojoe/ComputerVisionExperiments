// ConsoleApplication1.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include <iostream>
#include <thread>

using namespace std;

#define n_2 1000

void sequentialMerge(int *arrayA, int* arrayB, int* resultArray);
void isSorted_(int *array, int size);
void call_from_thread(int a, int b);
void insertIntoArray(int *startArray, int *destinationArray, int offset);
void printArray_(int *array, int size);

#define printArray(arr) printArray_((arr), sizeof(arr)/sizeof(arr[0]))
#define isSorted(arr) isSorted_((arr), sizeof(arr)/sizeof(arr[0]))

int main() {
	int arrayA[n_2];
	int arrayB[n_2];

	for (int i = 0; i < n_2; i++) {
		arrayA[i] = i * 2;
		arrayB[i] = (i * 2) + 1;
	}

	// Sequential
	cout << "SEQUENTIAL" << endl;
	int resultSequential[n_2 * 2];
	sequentialMerge(arrayA, arrayB, resultSequential);
	// printArray(resultSequential); #Debug
	isSorted(resultSequential);

	// Parallel
	cout << "PARALLEL" << endl;
	int resultParallel[n_2 * 2];
	thread t1(insertIntoArray, arrayA, resultParallel, 0);
	thread t2(insertIntoArray, arrayB, resultParallel, 1);
	t1.join();
	t2.join();
	// printArray(resultParallel); 
	isSorted(resultParallel);

	return 0;
}

void isSorted_(int *array, int size) {
	bool sorted = true;
	for (int i = 1; i < size; i++) {
		if (array[i] < array[i - 1]) {
			sorted = false;
		}
	}
	cout << "Sorted " << sorted << endl;
}

void sequentialMerge(int *arrayA, int* arrayB, int* resultArray) {
	for (int i = 0; i < n_2; i++) {
		resultArray[i * 2] = arrayA[i];
		resultArray[(i * 2) + 1] = arrayB[i];
	}
}

void call_from_thread(int a, int b) {
	cout << "Hello" << a << b << endl;
}

void insertIntoArray(int *startArray, int *destinationArray, int offset) {
	for (int i = 0; i < n_2; i++) {
		destinationArray[(i * 2) + offset] = startArray[i];
	}
}

void printArray_(int *array, int size) {
	for (int i = 0; i < size; i++) {
		cout << array[i] << " ";
	}
	cout << endl;
}

