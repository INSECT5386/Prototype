import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import io

# --- 설정 (사마귀 픽셀 이미지 생성용) ---
IMG_SIZE = 32          # 이미지 크기 (32x32) - 빠르고 직관적인 초기 수렴을 위해 작게 설정
POPULATION_SIZE = 50   # 한 세대의 총 개체 수 (부모 + 자식)
COLS = 15             # 결과 화면 표시 열 개수
ROWS = 10              # 결과 화면 표시 행 개수 (POPULATION_SIZE = COLS * ROWS)
MUTATION_RATE = 0.1   # 돌연변이 확률 (10%)
MUTATION_STRENGTH = 1.0 # 돌연변이 강도 (픽셀 값 변화 범위)

# --- 유전자(이미지) 표현 ---
def create_individual():
    """무작위 픽셀 값을 가진 단일 이미지를 생성합니다."""
    return np.random.uniform(0, 1, (IMG_SIZE, IMG_SIZE)) # 0~1 사이의 부동소수점

def create_population(size):
    """지정된 수만큼의 이미지 팝업레이션을 생성합니다."""
    return [create_individual() for _ in range(size)]

# --- 평가 (인간 개입) 및 선택 ---
def display_and_select(population, generation):
    """현재 세대의 이미지를 격자로 보여주고 사용자에게 선택을 받습니다."""
    fig = plt.figure(figsize=(COLS, ROWS))
    fig.suptitle(f"Generation {generation} - Select Mantis-like Candidates (by index)", fontsize=16, y=0.98)
    
    gs = gridspec.GridSpec(ROWS, COLS, wspace=0.1, hspace=0.3) # 간격 조정

    for i in range(len(population)):
        ax = plt.subplot(gs[i])
        ax.imshow(population[i], cmap='gray', vmin=0, vmax=1) # 흑백으로 표시
        ax.set_title(str(i), fontsize=8, pad=1)
        ax.axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1) # 사용자가 창을 닫을 때까지 대기

    print(f"\n--- Generation {generation} Evaluation ---")
    print("사마귀/동물 형태에 가깝다고 생각되는 이미지 번호들을 입력하세요.")
    print("예) 5, 23, 78, 112 (쉼표로 구분)")
    
    while True:
        try:
            indices_str = input("선택할 번호: ")
            if not indices_str:
                print("하나 이상 선택해야 합니다.")
                continue
            selected_indices = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
            if all(0 <= i < len(population) for i in selected_indices):
                break
            else:
                print(f"0부터 {len(population)-1} 사이의 번호만 입력하세요.")
        except ValueError:
            print("숫자와 쉼표만 입력하세요.")

    return [population[i] for i in selected_indices] # 선택된 부모들

# --- 번식 (교차 및 돌연변이) ---
def reproduce(parents, population_size):
    """선택된 부모들을 기반으로 새로운 자식 세대를 생성합니다."""
    new_population = []
    num_parents = len(parents)

    # 1. 엘리트주의 (Elitism): 부모는 그대로 다음 세대로 복제
    new_population.extend(parents)
    
    # 2. 자식 생성 (교차 및 돌연변이)
    while len(new_population) < population_size:
        # 부모 무작위 선택
        p1_idx = np.random.randint(0, num_parents)
        p2_idx = np.random.randint(0, num_parents)
        parent1 = parents[p1_idx]
        parent2 = parents[p2_idx]

        # 교차 (Crossover) - 균등 교차
        child = np.copy(parent1)
        mask = np.random.rand(IMG_SIZE, IMG_SIZE) < 0.5
        child[mask] = parent2[mask]

        # 돌연변이 (Mutation) - 가우시안 노이즈 추가
        if np.random.rand() < MUTATION_RATE:
            noise = np.random.normal(0, MUTATION_STRENGTH, (IMG_SIZE, IMG_SIZE))
            child = child + noise
            child = np.clip(child, 0, 1) # 0~1 범위를 벗어나지 않게 자름

        new_population.append(child)

    return new_population[:population_size] # 150개로 자름

# --- 메인 루프 ---
if __name__ == "__main__":
    print("=== Interactive Evolutionary Mantis Generator ===")
    
    # 1단계: 초기 팝업레이션 생성
    population = create_population(POPULATION_SIZE)
    generation = 0

    while True:
        generation += 1
        
        # 2~3단계: 평가 및 부모 선택
        selected_parents = display_and_select(population, generation)
        
        # 4단계: 번식 (자식 생성)
        population = reproduce(selected_parents, POPULATION_SIZE)
        
        # 계속 진행할지 확인
        cont = input(f"Generation {generation} 완료. 다음 세대로 진행할까요? (y/n): ")
        if cont.lower() != 'y':
            print("진화 프로세스를 종료합니다.")
            break
