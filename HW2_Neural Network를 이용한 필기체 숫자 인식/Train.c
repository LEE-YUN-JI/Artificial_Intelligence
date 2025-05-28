#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 784 // 28*28
#define NUM_TRAIN 60000
// Size of batch images, 온라인 모드로 구현되어 사용하지 않음
// # of hidden layers: 제약 없음, 사용자가 입력하도록 구현되어 있음
// # of nodes in hidden layers: 제약 없음, 사용자가 입력하도록 구현되어 있음
#define OUTPUT_NODES 10
#define ACTIVATION_FUNCTION(x) (1.0 / (1.0 + exp(-(x)))) // Activation Function: Sigmoid

#define LEARNING_RATE 0.1
#define EPOCHS 20

#define TRAIN_DATA_ADD "../../mnist_raw/training"
unsigned char train_image[NUM_TRAIN][INPUT_NODES];
unsigned char train_label[NUM_TRAIN];

typedef struct {
    int num_layers;
    int* layer_sizes;
    double** weights;
    double** activations;
    double** errors;
} NeuralNetwork;

NeuralNetwork create_network(int num_layers, int* layer_sizes);
void load_raw_image(const char* filename, unsigned char* image);
int load_mnist_raw(const char* data_dir, unsigned char(*images)[INPUT_NODES], unsigned char* labels);
void free_network(NeuralNetwork* network);
void forward_propagation(NeuralNetwork* network, double* input);
void backward_propagation(NeuralNetwork* network, int target, double learning_rate);
int predict(NeuralNetwork* network, double* input);
double calculate_accuracy(NeuralNetwork* network, unsigned char(*images)[INPUT_NODES], unsigned char* labels, int num_images);
void save_network(NeuralNetwork* network, const char* filename);
void shuffle_data(unsigned char(*images)[INPUT_NODES], unsigned char* labels, int n);

int main() {
    srand(time(NULL));
    int train_samples = load_mnist_raw(TRAIN_DATA_ADD, train_image, train_label);

    int num_hidden_layers;
    printf("Enter the number of hidden layers: ");
    scanf("%d", &num_hidden_layers);

    int num_layers = num_hidden_layers + 2;
    int* layer_sizes = (int*)malloc(num_layers * sizeof(int));
    layer_sizes[0] = INPUT_NODES;
    layer_sizes[num_layers - 1] = OUTPUT_NODES;

    for (int i = 1; i <= num_hidden_layers; i++) {
        printf("Enter the number of nodes for hidden layer %d: ", i);
        scanf("%d", &layer_sizes[i]);
    }

    NeuralNetwork network = create_network(num_layers, layer_sizes);

    clock_t start, finish;
    double duration;
    start = clock();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle_data(train_image, train_label, NUM_TRAIN);
        for (int i = 0; i < NUM_TRAIN; i++) {
            double input[INPUT_NODES];
            int target = train_label[i];

            for (int j = 0; j < INPUT_NODES; j++) {
            input[j] = train_image[i][j] / 255.0;
            }

            forward_propagation(&network, input);
            backward_propagation(&network, target, LEARNING_RATE);
    }

    double accuracy = calculate_accuracy(&network, train_image, train_label, NUM_TRAIN);
    printf("Epoch %d: Accuracy = %.2f%%\n", epoch + 1, accuracy * 100);
    }

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("Training 소요 시간: %.2lf sec", duration);

    char filename[100];
    snprintf(filename, sizeof(filename), "../../trained_network_%d.txt", num_hidden_layers);

    save_network(&network, filename);
    free_network(&network);
    free(layer_sizes);

    return 0;
}

NeuralNetwork create_network(int num_layers, int* layer_sizes) {
    NeuralNetwork network = {
        .num_layers = num_layers,
        .layer_sizes = malloc(num_layers * sizeof(int)),
        .weights = malloc((num_layers - 1) * sizeof(double*)),
        .activations = malloc(num_layers * sizeof(double*)),
        .errors = malloc(num_layers * sizeof(double*))
    };
    memcpy(network.layer_sizes, layer_sizes, num_layers * sizeof(int));

    for (int i = 0; i < num_layers - 1; i++) {
        network.weights[i] = (double*)malloc(layer_sizes[i] * layer_sizes[i + 1] * sizeof(double));
        for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            network.weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int i = 0; i < num_layers; i++) {
        network.activations[i] = (double*)malloc(layer_sizes[i] * sizeof(double));
        network.errors[i] = (double*)malloc(layer_sizes[i] * sizeof(double));
    }
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

        while (sample_index < NUM_TRAIN) {
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
    for (int i = 0; i < network->num_layers - 1; i++) free(network->weights[i]);
    for (int i = 0; i < network->num_layers; i++) {
        free(network->activations[i]);
        free(network->errors[i]);
    }
    free(network->weights);
    free(network->activations);
    free(network->errors);
    free(network->layer_sizes);
}

void forward_propagation(NeuralNetwork* network, double* input) {
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
}

void backward_propagation(NeuralNetwork* network, int target, double learning_rate) {
    int output_layer = network->num_layers - 1;

    for (int j = 0; j < network->layer_sizes[output_layer]; j++) {
        double output = network->activations[output_layer][j];
        network->errors[output_layer][j] = (output - (j == target ? 1.0 : 0.0));
    }

    for (int i = output_layer - 1; i > 0; i--) {
        for (int j = 0; j < network->layer_sizes[i]; j++) {
            double output = network->activations[i][j];
            double error_sum = 0.0;

            for (int k = 0; k < network->layer_sizes[i + 1]; k++) {
                error_sum += network->weights[i][j * network->layer_sizes[i + 1] + k] * network->errors[i + 1][k];
            }
            network->errors[i][j] = output * (1.0 - output) * error_sum;
        }
    }

    for (int l = 0; l < network->num_layers - 1; l++) {
        for (int i = 0; i < network->layer_sizes[l]; i++) {
            for (int j = 0; j < network->layer_sizes[l + 1]; j++) {
                int weight_idx = i * network->layer_sizes[l + 1] + j;
                network->weights[l][weight_idx] -= learning_rate *
                    (network->activations[l][i] * network->errors[l + 1][j]);
            }
        }
    }
}

int predict(NeuralNetwork* network, double* input) {
    forward_propagation(network, input);
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

double calculate_accuracy(NeuralNetwork* network, unsigned char(*images)[INPUT_NODES], unsigned char* labels, int num_images) {
    int correct = 0;
    double input[INPUT_NODES];
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < INPUT_NODES; j++) input[j] = images[i][j] / 255.0;
        if (predict(network, input) == labels[i]) correct++;
    }
    return (double)correct / num_images;
}

void save_network(NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Could not open file for writing: %s\n", filename);
        return;
    }

    fprintf(file, "Number of layers: %d\n", network->num_layers);
    fprintf(file, "Layer sizes: ");
    for (int i = 0; i < network->num_layers; i++) {
        fprintf(file, "%d ", network->layer_sizes[i]);
    }
    fprintf(file, "\n\n");

    for (int i = 0; i < network->num_layers - 1; i++) {
        fprintf(file, "Weights for layer %d:\n", i + 1);
        for (int j = 0; j < network->layer_sizes[i]; j++) {
            for (int k = 0; k < network->layer_sizes[i + 1]; k++) {
                fprintf(file, "%.6f ", network->weights[i][j * network->layer_sizes[i + 1] + k]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void shuffle_data(unsigned char(*images)[INPUT_NODES], unsigned char* labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        unsigned char temp_image[INPUT_NODES];
        memcpy(temp_image, images[i], INPUT_NODES * sizeof(unsigned char));
        memcpy(images[i], images[j], INPUT_NODES * sizeof(unsigned char));
        memcpy(images[j], temp_image, INPUT_NODES * sizeof(unsigned char));

        unsigned char temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}
