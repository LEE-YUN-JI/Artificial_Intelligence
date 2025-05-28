#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 512
#define HEIGHT 512
#define INPUT_FILE "../../raw_image/I1.img"
#define OUTPUT_FILE "../../raw_image/I1"

int select(void);
unsigned char* load_image(const char* filename);
int rotate(int choice);
int save_image(const char* filename, unsigned char* img);

int main(void) {
    select();
}

int select(void) {
    int choice;
    printf("1. Clockwise 90 | 2. Clockwise 180 | 3. Clockwise 270 | 4. Mirror | 5. Flip \n회전 방향을 선택하세요: ");
    scanf("%d", &choice);

    if (choice >= 1 && choice <= 5) {
        return rotate(choice);
    }
    else {
        printf("잘못된 선택입니다.");
        return 1;
    }
}

unsigned char* load_image(const char* filename) {
    FILE* file_input = NULL;
    unsigned char* img = NULL;

    file_input = fopen(filename, "rb");
    if (file_input == NULL) {
        printf("파일이 존재하지 않습니다.\n");
        return NULL;
    }

    img = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
    if (img == NULL) {
        printf("메모리 할당 오류 발생\n");
        fclose(file_input);
        return NULL;
    }

    fread(img, sizeof(unsigned char), WIDTH * HEIGHT, file_input);
    fclose(file_input);

    return img;
}

int rotate(int choice) {
    unsigned char* img = NULL;
    unsigned char* img_rotate = NULL;

    img = load_image(INPUT_FILE);
    if (img == NULL) {
        return -1;
    }

    img_rotate = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
    if (img_rotate == NULL) {
        printf("메모리 할당 오류 발생\n");
        free(img);
        return -1;
    }

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int new_i, new_j;
            switch (choice) {
            case 1: // Clockwise 90
                new_i = j;
                new_j = HEIGHT - 1 - i;
                break;
            case 2: // Clockwise 180
                new_i = HEIGHT - 1 - i;
                new_j = WIDTH - 1 - j;
                break;
            case 3: // Clockwise 270
                new_i = WIDTH - 1 - j;
                new_j = i;
                break;
            case 4: // Mirror
                new_i = i;
                new_j = WIDTH - 1 - j;
                break;
            case 5: // Flip
                new_i = HEIGHT - 1 - i;
                new_j = j;
                break;
            }
            *(img_rotate + new_i * WIDTH + new_j) = *(img + i * WIDTH + j);
        }
    }

    char output_filename[256];
    switch (choice) {
    case 1: sprintf(output_filename, "%s_90.img", OUTPUT_FILE); break;
    case 2: sprintf(output_filename, "%s_180.img", OUTPUT_FILE); break;
    case 3: sprintf(output_filename, "%s_270.img", OUTPUT_FILE); break;
    case 4: sprintf(output_filename, "%s_mirror.img", OUTPUT_FILE); break;
    case 5: sprintf(output_filename, "%s_flip.img", OUTPUT_FILE); break;
    }

    if (save_image(output_filename, img_rotate) != 0) {
        printf("이미지 저장 실패\n");
    }

    free(img_rotate);
    free(img);

    return 0;
}

int save_image(const char* filename, unsigned char* img) {
    FILE* file_output = NULL;

    file_output = fopen(filename, "wb");
    if (file_output == NULL) {
        printf("파일을 열 수 없습니다.\n");
        return -1;
    }

    fwrite(img, sizeof(unsigned char), WIDTH * HEIGHT, file_output);
    fclose(file_output);

    return 0;
}