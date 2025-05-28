#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_NODES 784
#define OUTPUT_NODES 10
#define NUM_TEST 10000
#define ACTIVATION_FUNCTION(x) (1.0 / (1.0 + exp(-(x))))

typedef struct {
    int num_layers;
    int* layer_sizes;
    double** weights;
    double** activations;
} NeuralNetwork;

#define TEST_DATA_DIR "../../mnist_raw/testing"
unsigned char test_image[NUM_TEST][INPUT_NODES];
unsigned char test_label[NUM_TEST];

NeuralNetwork load_network(const char* filename);
void load_raw_image(const char* filename, unsigned char* image);
int load_mnist_raw(const char* data_dir, unsigned char(*images)[INPUT_NODES], unsigned char* labels);
void free_network(NeuralNetwork* network);
int predict(NeuralNetwork* network, double* input);

int main() {
    int test_samples = load_mnist_raw(TEST_DATA_DIR, test_image, test_label);

    int num_hidden_layers;
    printf("Enter the number of hidden layers to use: ");
    scanf("%d", &num_hidden_layers);

    char filename[100];
    snprintf(filename, sizeof(filename), "../../trained_network_%d.txt", num_hidden_layers);

    NeuralNetwork network = load_network(filename);

    int correct = 0;
    int digit_correct[10] = { 0 };
    int digit_total[10] = { 0 };

    for (int i = 0; i < NUM_TEST; i++) {
        double input[INPUT_NODES];
        for (int j = 0; j < INPUT_NODES; j++) {
            input[j] = test_image[i][j] / 255.0;
        }

        int prediction = predict(&network, input);
        int true_label = test_label[i];

        digit_total[true_label]++;

        if (prediction == true_label) {
            correct++;
            digit_correct[true_label]++;
        }
    }

    double overall_accuracy = (double)correct / NUM_TEST * 100;
    printf("Overall accuracy: %.2f%%\n", overall_accuracy);

    printf("\nAccuracy for each digit:\n");
    for (int i = 0; i < 10; i++) {
        double digit_accuracy = (double)digit_correct[i] / digit_total[i] * 100;
        printf("Digit %d: %.2f%% (%d/%d)\n", i, digit_accuracy, digit_correct[i], digit_total[i]);
    }

    free_network(&network);
    return 0;
}

NeuralNetwork load_network(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        exit(1);
    }

    NeuralNetwork network;
    fscanf(file, "Number of layers: %d\n", &network.num_layers);

    network.layer_sizes = (int*)malloc(network.num_layers * sizeof(int));
    fscanf(file, "Layer sizes: ");
    for (int i = 0; i < network.num_layers; i++) {
        fscanf(file, "%d ", &network.layer_sizes[i]);
    }

    network.weights = (double**)malloc((network.num_layers - 1) * sizeof(double*));
    network.activations = (double**)malloc(network.num_layers * sizeof(double*));

    for (int i = 0; i < network.num_layers - 1; i++) {
        network.weights[i] = (double*)malloc(network.layer_sizes[i] * network.layer_sizes[i + 1] * sizeof(double));
        network.activations[i] = (double*)malloc(network.layer_sizes[i] * sizeof(double));

        fscanf(file, "\nWeights for layer %*d:\n");
        for (int j = 0; j < network.layer_sizes[i]; j++) {
            for (int k = 0; k < network.layer_sizes[i + 1]; k++) {
                fscanf(file, "%lf ", &network.weights[i][j * network.layer_sizes[i + 1] + k]);
            }
        }
    }
    network.activations[network.num_layers - 1] = (double*)malloc(network.layer_sizes[network.num_layers - 1] * sizeof(double));

    fclose(file);
    return network;
}

void load_raw_image(const char* filename, unsigned char* image) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Warning: Could not open %s\n", filename);
        return;
    }

    fread(image, sizeof(unsigned char), INPUT_NODES, file);
    fclose(file);
}

int load_mnist_raw(const char* data_dir, unsigned char(*images)[INPUT_NODES], unsigned char* labels) {
    int sample_index = 0;

    for (int label = 0; label < OUTPUT_NODES; label++) {
        int img_index = 0;

        printf("[Loading] Label %d...\n", label);

        while (sample_index < NUM_TEST) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/%d/%d-%d.raw", data_dir, label, label, img_index);

            FILE* file = fopen(filename, "rb");
            if (!file) {
                break;
            }
            fclose(file);

            load_raw_image(filename, images[sample_index]);
            labels[sample_index] = (unsigned char)label;

            sample_index++;
            img_index++;
        }
    }

    printf("[Done] Loaded %d samples from %s\n", sample_index, data_dir);
    return sample_index;
}

void free_network(NeuralNetwork* network) {
    for (int i = 0; i < network->num_layers - 1; i++) {
        free(network->weights[i]);
        free(network->activations[i]);
    }
    free(network->activations[network->num_layers - 1]);
    free(network->weights);
    free(network->activations);
    free(network->layer_sizes);
}

int predict(NeuralNetwork* network, double* input) {
    memcpy(network->activations[0], input, network->layer_sizes[0] * sizeof(double));

    for (int i = 1; i < network->num_layers; i++) {
        for (int j = 0; j < network->layer_sizes[i]; j++) {
            double sum = 0;
            for (int k = 0; k < network->layer_sizes[i - 1]; k++) {
                sum += network->activations[i - 1][k] * network->weights[i - 1][k * network->layer_sizes[i] + j];
            }
            network->activations[i][j] = ACTIVATION_FUNCTION(sum);
        }
    }

    int output_layer = network->num_layers - 1;
    int max_index = 0;
    double max_value = network->activations[output_layer][0];

    for (int i = 1; i < network->layer_sizes[output_layer]; i++) {
        if (network->activations[output_layer][i] > max_value) {
            max_value = network->activations[output_layer][i];
            max_index = i;
        }
    }
    return max_index;
}