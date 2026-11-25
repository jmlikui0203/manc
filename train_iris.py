from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier


def save_loss_plot(features, labels, output_path: Path) -> None:
    depths = range(1, 5)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    training_losses = []
    validation_losses = []

    for depth in depths:
        scores = cross_validate(
            DecisionTreeClassifier(max_depth=depth, random_state=42),
            features,
            labels,
            cv=folds,
            scoring="neg_log_loss",
            return_train_score=True,
        )
        training_losses.append(-scores["train_score"].mean())
        validation_losses.append(-scores["test_score"].mean())

    figure = Figure(figsize=(7, 4), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(depths, training_losses, "o-", label="Training")
    axes.plot(depths, validation_losses, "s-", label="Validation")
    axes.set(
        title="Iris decision tree | 5-fold cross-validation",
        xlabel="Maximum tree depth",
        ylabel="Mean log loss (lower is better)",
        xticks=list(depths),
    )
    axes.grid(axis="y", alpha=0.25)
    axes.legend()
    canvas.print_png(output_path)


def main() -> None:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"训练样本：{len(y_train)}，测试样本：{len(y_test)}")
    print(f"测试集准确率：{accuracy_score(y_test, predictions):.1%}")
    print("类别顺序：" + "、".join(iris.target_names))
    print("混淆矩阵（行是真实类别，列是预测类别）")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    plot_path = Path(__file__).with_name("iris_loss.png")
    save_loss_plot(x_train, y_train, plot_path)
    print(f"损失图：{plot_path}")


if __name__ == "__main__":
    main()
