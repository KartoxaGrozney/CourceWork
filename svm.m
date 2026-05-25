%% svm_radius_local_lines.m
% Улучшенная версия SVM-визуализации через радиус соседства.
%
% Логика сохранена как в вашем варианте:
%   1) читаем classification.csv: x1, x2, class_id;
%   2) соседние классы ищем через радиус R;
%   3) для соседних классов строим линейные SVM-гиперплоскости.
%
% Улучшение:
%   - SVM строится локально около общей границы классов, а не по всем точкам пары;
%   - если одна пара классов имеет несколько раздельных участков границы,
%     для каждого участка строится свой короткий отрезок;
%   - линии не протягиваются через весь рисунок;
%   - цвета корректно работают при любом количестве классов.

clear; clc; close all;

%% 1. ПАРАМЕТРЫ
dataFile = 'classification.csv';

% Радиус соседства: если точки разных классов находятся ближе R,
% классы считаются соседними на данном участке.
R = 0.055

% Минимальное количество близких пар точек для построения линии.
minNeighborPairs = 3;

% Радиус объединения близких пар в один участок границы.
% Если линий слишком много — увеличьте componentRadius.
% Если линии слишком длинные — уменьшите componentRadius.
componentRadius = 2.0 * R;

% Радиус локального обучения SVM вокруг участка границы.
% Чем больше значение, тем больше точек участвует в построении прямой.
trainRadius = 2.5 * R;

% Насколько длинным рисовать отрезок гиперплоскости.
drawMargin = 0.75 * R;

% Параметры SVM
C = 1e6;             % большое C ~= hard-margin
tol = 1e-3;
maxPasses = 15;
alphaTol = 1e-6;

% Ограничения на размер локальной обучающей выборки
maxLocalPointsPerClass = 150;

% Настройки графика
pointSize = 10;
lineWidth = 1.2;
rng(42);

%% 2. ЧТЕНИЕ ДАННЫХ
data = readmatrix(dataFile);

if isempty(data) || size(data,2) < 3
    error('Файл classification.csv должен содержать 3 столбца: x1, x2, class_id.');
end

data = data(:,1:3);
data = data(all(~isnan(data),2), :);

x = data(:,1);
y = data(:,2);
labels = data(:,3);

classes = unique(labels, 'stable');
numClasses = numel(classes);

fprintf('Загружено точек: %d\n', numel(labels));
fprintf('Количество классов: %d\n', numClasses);

colors = buildColors(numClasses);

%% 3. ПОДГОТОВКА ГРАФИКА
figure('Color','w'); hold on; grid on;

% Для большого числа классов легенду не строим, иначе MATLAB выдаёт warning.
if numClasses <= 30
    scatterHandles = gobjects(numClasses,1);
    for i = 1:numClasses
        cl = classes(i);
        idx = labels == cl;
        scatterHandles(i) = scatter(x(idx), y(idx), pointSize, ...
            'MarkerFaceColor', colors(i,:), ...
            'MarkerEdgeColor', colors(i,:), ...
            'DisplayName', sprintf('Class %g', cl));
    end
else
    labelIndex = zeros(size(labels));
    for i = 1:numClasses
        labelIndex(labels == classes(i)) = i;
    end
    scatter(x, y, pointSize, labelIndex, 'filled', 'MarkerEdgeColor', 'none');
    colormap(colors);
    cb = colorbar;
    cb.Label.String = 'Порядковый номер класса';
    caxis([1 numClasses]);
end

xlabel('x_1');
ylabel('x_2');
title('Классы и локальные разделяющие гиперплоскости SVM');
axis equal;

xMin = min(x); xMax = max(x);
yMin = min(y); yMax = max(y);
xlim([xMin, xMax]);
ylim([yMin, yMax]);

%% 4. ПОИСК СОСЕДНИХ КЛАССОВ ЧЕРЕЗ РАДИУС И ПОСТРОЕНИЕ ЛОКАЛЬНЫХ SVM
totalSegments = 0;
totalNeighborClassPairs = 0;

for ii = 1:numClasses-1
    ci = classes(ii);
    idxiGlobal = find(labels == ci);
    Xi = [x(idxiGlobal), y(idxiGlobal)];

    for jj = ii+1:numClasses
        cj = classes(jj);
        idxjGlobal = find(labels == cj);
        Xj = [x(idxjGlobal), y(idxjGlobal)];

        % Находим близкие пары точек двух классов.
        [pairsI, pairsJ, midpoints] = findClosePairs(Xi, Xj, R);

        if size(midpoints,1) < minNeighborPairs
            continue;
        end

        totalNeighborClassPairs = totalNeighborClassPairs + 1;

        % Разбиваем близкие пары на локальные участки общей границы.
        components = splitMidpointsIntoComponents(midpoints, componentRadius);

        for compId = 1:numel(components)
            compIdx = components{compId};

            if numel(compIdx) < minNeighborPairs
                continue;
            end

            compMid = midpoints(compIdx, :);

            % Локальные точки для обучения SVM около этого участка границы.
            localI = selectLocalPoints(Xi, compMid, trainRadius);
            localJ = selectLocalPoints(Xj, compMid, trainRadius);

            if size(localI,1) < 2 || size(localJ,1) < 2
                continue;
            end

            localI = sampleRows(localI, maxLocalPointsPerClass);
            localJ = sampleRows(localJ, maxLocalPointsPerClass);

            Xtrain = [localI; localJ];
            ytrain = [-ones(size(localI,1),1); +ones(size(localJ,1),1)];

            % Если локальные точки почти неразделимы или плохо обусловлены,
            % SMO может не построить устойчивую прямую. В таком случае пропускаем участок.
            try
                model = trainBinaryLinearSVM_SMO(Xtrain, ytrain, C, tol, maxPasses, alphaTol);
            catch
                continue;
            end

            % Рисуем не бесконечную прямую, а короткий отрезок около участка границы.
            [p1, p2, ok] = svmSegmentNearBoundary(model, compMid, drawMargin);

            if ok
                plot([p1(1), p2(1)], [p1(2), p2(2)], 'k-', ...
                    'LineWidth', lineWidth, ...
                    'HandleVisibility','off');
                totalSegments = totalSegments + 1;
            end
        end
    end
end

fprintf('Найдено соседних пар классов: %d\n', totalNeighborClassPairs);
fprintf('Построено локальных SVM-отрезков: %d\n', totalSegments);

if numClasses <= 30
    legend(scatterHandles, 'Location', 'eastoutside');
end

hold off;

%% ===== ЛОКАЛЬНЫЕ ФУНКЦИИ =====

function colors = buildColors(numClasses)
    baseColors = [
        0.00 0.45 0.74
        0.85 0.33 0.10
        0.93 0.69 0.13
        0.49 0.18 0.56
        0.30 0.75 0.93
        0.47 0.67 0.19
        0.64 0.08 0.18
        0.00 0.00 0.00
        0.75 0.75 0.75
        1.00 0.00 0.00
        0.00 0.50 0.00
        0.00 0.00 0.50
    ];

    if numClasses <= size(baseColors,1)
        colors = baseColors(1:numClasses,:);
    else
        colors = turbo(numClasses);
    end
end

function Xs = sampleRows(X, maxN)
    n = size(X,1);
    if n <= maxN
        Xs = X;
    else
        idx = randperm(n, maxN);
        Xs = X(idx,:);
    end
end

function [pairsI, pairsJ, midpoints] = findClosePairs(Xi, Xj, R)
    pairsI = [];
    pairsJ = [];
    midpoints = [];

    R2 = R^2;

    for p = 1:size(Xi,1)
        dx = Xj(:,1) - Xi(p,1);
        dy = Xj(:,2) - Xi(p,2);
        d2 = dx.^2 + dy.^2;

        nearJ = find(d2 <= R2);

        if ~isempty(nearJ)
            pairsI = [pairsI; repmat(p, numel(nearJ), 1)]; %#ok<AGROW>
            pairsJ = [pairsJ; nearJ(:)]; %#ok<AGROW>
            mids = 0.5 * (repmat(Xi(p,:), numel(nearJ), 1) + Xj(nearJ,:));
            midpoints = [midpoints; mids]; %#ok<AGROW>
        end
    end
end

function components = splitMidpointsIntoComponents(M, radius)
    n = size(M,1);
    if n == 0
        components = {};
        return;
    end

    visited = false(n,1);
    components = {};
    r2 = radius^2;

    for i = 1:n
        if visited(i)
            continue;
        end

        queue = i;
        visited(i) = true;
        comp = i;

        while ~isempty(queue)
            q = queue(1);
            queue(1) = [];

            dx = M(:,1) - M(q,1);
            dy = M(:,2) - M(q,2);
            d2 = dx.^2 + dy.^2;

            neigh = find(d2 <= r2 & ~visited);

            if ~isempty(neigh)
                visited(neigh) = true;
                queue = [queue; neigh(:)]; %#ok<AGROW>
                comp = [comp; neigh(:)]; %#ok<AGROW>
            end
        end

        components{end+1} = comp; %#ok<AGROW>
    end
end

function Xloc = selectLocalPoints(X, M, radius)
    if isempty(X) || isempty(M)
        Xloc = zeros(0,2);
        return;
    end

    r2 = radius^2;
    keep = false(size(X,1),1);

    for k = 1:size(M,1)
        dx = X(:,1) - M(k,1);
        dy = X(:,2) - M(k,2);
        keep = keep | (dx.^2 + dy.^2 <= r2);
    end

    Xloc = X(keep,:);
end

function model = trainBinaryLinearSVM_SMO(X, y, C, tol, maxPasses, alphaTol)
    [M, n] = size(X);

    if M < 4 || numel(unique(y)) < 2
        error('Недостаточно точек для бинарного SVM.');
    end

    % Стандартизация признаков для устойчивости SMO.
    mu = mean(X,1);
    sigma = std(X,0,1);
    sigma(sigma < 1e-12) = 1;

    Xs = (X - mu) ./ sigma;
    y = y(:);

    alpha = zeros(M,1);
    b = 0;
    K = Xs * Xs';

    passes = 0;

    while passes < maxPasses
        numChanged = 0;

        for i = 1:M
            f_i = sum(alpha .* y .* K(:,i)) + b;
            E_i = f_i - y(i);

            if (y(i)*E_i < -tol && alpha(i) < C) || ...
               (y(i)*E_i >  tol && alpha(i) > 0)

                j = randi(M);
                while j == i
                    j = randi(M);
                end

                f_j = sum(alpha .* y .* K(:,j)) + b;
                E_j = f_j - y(j);

                ai_old = alpha(i);
                aj_old = alpha(j);

                if y(i) ~= y(j)
                    L = max(0, aj_old - ai_old);
                    H = min(C, C + aj_old - ai_old);
                else
                    L = max(0, ai_old + aj_old - C);
                    H = min(C, ai_old + aj_old);
                end

                if abs(L - H) < eps
                    continue;
                end

                eta = 2*K(i,j) - K(i,i) - K(j,j);

                if eta >= 0
                    continue;
                end

                alpha(j) = aj_old - y(j)*(E_i - E_j)/eta;
                alpha(j) = min(H, max(L, alpha(j)));

                if abs(alpha(j) - aj_old) < 1e-12
                    alpha(j) = aj_old;
                    continue;
                end

                alpha(i) = ai_old + y(i)*y(j)*(aj_old - alpha(j));

                b1 = b - E_i ...
                    - y(i)*(alpha(i)-ai_old)*K(i,i) ...
                    - y(j)*(alpha(j)-aj_old)*K(i,j);

                b2 = b - E_j ...
                    - y(i)*(alpha(i)-ai_old)*K(i,j) ...
                    - y(j)*(alpha(j)-aj_old)*K(j,j);

                if alpha(i) > 0 && alpha(i) < C
                    b = b1;
                elseif alpha(j) > 0 && alpha(j) < C
                    b = b2;
                else
                    b = 0.5*(b1 + b2);
                end

                numChanged = numChanged + 1;
            end
        end

        if numChanged == 0
            passes = passes + 1;
        else
            passes = 0;
        end
    end

    w = Xs' * (alpha .* y);

    sv = find(alpha > alphaTol);
    if ~isempty(sv)
        b = mean(y(sv) - Xs(sv,:)*w);
    end

    % Перевод прямой из стандартизированных координат в исходные:
    % ((x-mu)./sigma)*w + b = nOrig*x + bOrig
    nOrig = [w(1)/sigma(1); w(2)/sigma(2)];
    bOrig = b - (w(1)*mu(1)/sigma(1) + w(2)*mu(2)/sigma(2));

    if norm(nOrig) < 1e-12
        error('Нулевая нормаль гиперплоскости.');
    end

    model.wStd = w;
    model.bStd = b;
    model.mu = mu;
    model.sigma = sigma;
    model.nOrig = nOrig;
    model.bOrig = bOrig;
end

function [p1, p2, ok] = svmSegmentNearBoundary(model, midpoints, margin)
    n = model.nOrig(:);
    b = model.bOrig;

    normN = norm(n);
    if normN < 1e-12 || isempty(midpoints)
        p1 = [NaN, NaN];
        p2 = [NaN, NaN];
        ok = false;
        return;
    end

    n = n / normN;
    b = b / normN;

    % Направление прямой.
    d = [-n(2); n(1)];

    % Точка на прямой n'*p + b = 0.
    p0 = -b * n;

    % Проецируем midpoints на направление прямой.
    t = midpoints * d;

    t1 = min(t) - margin;
    t2 = max(t) + margin;

    if abs(t2 - t1) < 1e-9
        t1 = t1 - margin;
        t2 = t2 + margin;
    end

    p1 = (p0 + t1*d).';
    p2 = (p0 + t2*d).';

    ok = all(isfinite(p1)) && all(isfinite(p2));
end
