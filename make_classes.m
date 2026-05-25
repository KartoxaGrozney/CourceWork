clear; clc;

params.T = 10;              
params.tau = 7;            
params.h = 1;              
params.nPoints = 20000;      

params.xMin = -2;           
params.xMax =  2;           
params.seed = 42;          

params.discretization = 'exact';     
params.modeTol = 1e-7;               
params.simplexTol = 1e-9;           
params.maxSimplexIter = 20000;       

params.structureTol = 1e-5;

% решение
classificationMatrix = buildClassificationMatrix(params);

% сохранение
writematrix(classificationMatrix, 'classification.csv');

fprintf('\nГотово. Создан один файл: classification.csv\n');
fprintf('Формат: x1, x2, class_id\n');
fprintf('Количество точек с решением: %d\n', size(classificationMatrix, 1));
fprintf('Количество классов: %d\n', numel(unique(classificationMatrix(:,3))));

% график

plotClassification(classificationMatrix);


function classificationMatrix = buildClassificationMatrix(params)
    validateParams(params);
    rng(params.seed);

    x1Values = params.xMin + (params.xMax - params.xMin) * rand(params.nPoints, 1);
    x2Values = params.xMin + (params.xMax - params.xMin) * rand(params.nPoints, 1);

    solvedX1 = zeros(params.nPoints, 1);
    solvedX2 = zeros(params.nPoints, 1);
    solvedClass = zeros(params.nPoints, 1);
    solvedCount = 0;

    classKeys = strings(0, 1);
    classIds = zeros(0, 1);
    nextClassId = 1;

    for i = 1:params.nPoints
        x0 = [x1Values(i); x2Values(i)];

        result = solveOptimalControlAtPoint(x0, params);

        if result.hasSolution
            % Формирование класса 
            classKey = makeClassKeyByArticleStructure(result.u, params);

            idxClass = find(classKeys == classKey, 1);
            if isempty(idxClass)
                classKeys(end+1, 1) = classKey; %#ok<AGROW>
                classIds(end+1, 1) = nextClassId; %#ok<AGROW>
                assignedClass = nextClassId;
                nextClassId = nextClassId + 1;
            else
                assignedClass = classIds(idxClass);
            end

            solvedCount = solvedCount + 1;
            solvedX1(solvedCount) = x0(1);
            solvedX2(solvedCount) = x0(2);
            solvedClass(solvedCount) = assignedClass;
        end

        if mod(i, 100) == 0 || i == params.nPoints
            fprintf('Обработано %d из %d точек...\n', i, params.nPoints);
        end
    end

    classificationMatrix = [ ...
        solvedX1(1:solvedCount), ...
        solvedX2(1:solvedCount), ...
        solvedClass(1:solvedCount) ...
    ];
end

function validateParams(params)
    if params.T <= params.tau
        error('Должно быть T > tau.');
    end

    Nreal = (params.T - params.tau) / params.h;
    if abs(Nreal - round(Nreal)) > 1e-10
        error('(T - tau) / h должно быть целым числом.');
    end

    if params.nPoints <= 0
        error('Количество точек должно быть положительным.');
    end

    if params.xMin >= params.xMax
        error('Должно быть xMin < xMax.');
    end
end

function plotClassification(classificationMatrix)
    if isempty(classificationMatrix)
        warning('Нет точек с решением, график не построен.');
        return;
    end

    x1 = classificationMatrix(:, 1);
    x2 = classificationMatrix(:, 2);
    classId = classificationMatrix(:, 3);

    figure;
    scatter(x1, x2, 14, classId, 'filled');
    grid on;
    axis equal;
    xlabel('x_1');
    ylabel('x_2');
    title('Разделение точек фазовой плоскости на классы');
    colorbar;
end

function result = solveOptimalControlAtPoint(x0, params)
    N = round((params.T - params.tau) / params.h);

    [Aeq, beq, c, F, g] = buildControlLP(x0, params, N);

    opts.tol = params.simplexTol;
    opts.maxIter = params.maxSimplexIter;

    [y, info] = simplexTwoPhase(Aeq, beq, c, opts);

    result = struct();
    result.hasSolution = false;
    result.u = [];

    if ~info.success
        return;
    end

    p = y(1:N);
    m = y(N+1:2*N);
    u = p - m;
    u(abs(u) < params.modeTol) = 0;

    % Проверка попадания в конечное состояние.
    x = x0(:);
    for k = 1:N
        x = F * x + g * u(k);
    end

    if norm(x) > 1e-5
        return;
    end

    result.hasSolution = true;
    result.u = u(:);
end

function [Aeq, beq, c, F, g] = buildControlLP(x0, params, N)
    h = params.h;

    if strcmpi(params.discretization, 'exact')
        F = [cos(h),  sin(h);
            -sin(h),  cos(h)];
        g = [1 - cos(h);
             sin(h)];
    elseif strcmpi(params.discretization, 'euler')
        A = [0 1; -1 0];
        b = [0; 1];
        F = eye(2) + h * A;
        g = h * b;
    else
        error('Неизвестный тип дискретизации: %s', params.discretization);
    end

    nVars = 3 * N;
    nEq = 2 + N;

    Aeq = zeros(nEq, nVars);
    beq = zeros(nEq, 1);

    % Условие x_N = 0:
    % F^N x0 + sum_{k=1}^N F^{N-k} g u_k = 0.
    beq(1:2) = -(F^N) * x0(:);

    for k = 1:N
        coeff = (F^(N-k)) * g;
        Aeq(1:2, k) = coeff;        % p_k
        Aeq(1:2, N + k) = -coeff;   % m_k
    end

    % Ограничение |u_k| <= 1 через p_k + m_k + s_k = 1.
    for k = 1:N
        row = 2 + k;
        Aeq(row, k) = 1;
        Aeq(row, N + k) = 1;
        Aeq(row, 2*N + k) = 1;
        beq(row) = 1;
    end

    % Целевая функция h * sum(|u_k|) = h * sum(p_k + m_k).
    c = zeros(nVars, 1);
    c(1:N) = h;
    c(N+1:2*N) = h;
end

function key = makeClassKeyByArticleStructure(u, params)
    tau = params.tau;
    h = params.h;
    structureTol = params.structureTol;
    modeTol = params.modeTol;

    u = u(:);
    N = numel(u);

    % s_k = tau + (k-1)h, k = 1,...,N.
    timeGrid = tau + (0:N-1).' * h;

    % Опорные точки
    isSupport = abs(u) < 1 - structureTol;
    T_op = timeGrid(isSupport);

    U_op_modes = zeros(sum(isSupport), 1);
    supportValues = u(isSupport);
    for q = 1:numel(supportValues)
        U_op_modes(q) = signWithTol(supportValues(q), modeTol);
    end

    % Неопорные точки переключения
    T_n0 = [];
    for k = 2:N
        if u(k-1) * u(k) < -modeTol
            T_n0(end+1, 1) = timeGrid(k); %#ok<AGROW>
        end
    end

    % gamma
    if ~isSupport(1)
        gamma = signWithTol(u(1), modeTol);
    elseif N >= 2
        gamma = signWithTol(u(2), modeTol);
    else
        gamma = 0;
    end

    TOpText = vectorToKeyText(T_op);
    UOpText = vectorToKeyText(U_op_modes);
    TN0Text = vectorToKeyText(T_n0);

    key = sprintf('tau=%g|gamma=%+d|Top=%s|Uop=%s|Tn0=%s', ...
        tau, gamma, TOpText, UOpText, TN0Text);
end

function s = signWithTol(value, tol)
    if value > tol
        s = 1;
    elseif value < -tol
        s = -1;
    else
        s = 0;
    end
end

function txt = vectorToKeyText(v)
    if isempty(v)
        txt = 'none';
        return;
    end

    parts = strings(1, numel(v));
    for i = 1:numel(v)
        parts(i) = sprintf('%g', v(i));
    end
    txt = char(strjoin(parts, '_'));
end

function [x, info] = simplexTwoPhase(A, b, c, opts)
    tol = opts.tol;
    maxIter = opts.maxIter;

    [m, n] = size(A);

    for i = 1:m
        if b(i) < 0
            A(i,:) = -A(i,:);
            b(i) = -b(i);
        end
    end

    A1 = [A, eye(m)];
    c1 = [zeros(n,1); ones(m,1)];
    basis = (n+1:n+m).';

    [~, basis1, obj1, status1] = simplexCore(A1, b, c1, basis, tol, maxIter);

    info = struct();
    info.success = false;
    info.status = "UNKNOWN";
    info.phase1Objective = obj1;

    if ~strcmp(status1, "OPTIMAL")
        x = [];
        info.status = "PHASE1_" + status1;
        return;
    end

    if obj1 > 1e-7
        x = [];
        info.status = "INFEASIBLE";
        return;
    end

    [basis2, ok] = removeArtificialFromBasis(A1, A, basis1, n, tol);

    if ~ok
        x = [];
        info.status = "CANNOT_REMOVE_ARTIFICIAL_BASIS";
        return;
    end

    [x2, ~, obj2, status2] = simplexCore(A, b, c, basis2, tol, maxIter);

    if ~strcmp(status2, "OPTIMAL")
        x = [];
        info.status = "PHASE2_" + status2;
        return;
    end

    x = x2;
    info.success = true;
    info.status = "OPTIMAL";
    info.objective = obj2;
end

function [x, basis, obj, status] = simplexCore(A, b, c, basis, tol, maxIter)
    [m, n] = size(A);
    basis = basis(:);

    status = "UNKNOWN";
    x = zeros(n, 1);
    obj = NaN;

    for iter = 1:maxIter %#ok<NASGU>
        B = A(:, basis);

        if rcond(B) < 1e-14
            status = "SINGULAR_BASIS";
            return;
        end

        xB = B \ b;

        if any(xB < -1e-7)
            status = "INFEASIBLE_BASIS";
            return;
        end
        xB(abs(xB) < tol) = 0;

        cB = c(basis);
        lambda = B' \ cB;
        reduced = c - A' * lambda;

        nonBasis = setdiff((1:n).', basis, 'stable');
        [minReduced, pos] = min(reduced(nonBasis));

        if minReduced >= -tol
            x = zeros(n, 1);
            x(basis) = xB;
            obj = c' * x;
            status = "OPTIMAL";
            return;
        end

        entering = nonBasis(pos);
        d = B \ A(:, entering);

        if all(d <= tol)
            status = "UNBOUNDED";
            return;
        end

        ratios = inf(m, 1);
        positive = d > tol;
        ratios(positive) = xB(positive) ./ d(positive);

        minRatio = min(ratios);
        leaveCandidates = find(abs(ratios - minRatio) <= 100 * tol);

        [~, bestLocal] = min(basis(leaveCandidates));
        leavingRow = leaveCandidates(bestLocal);

        basis(leavingRow) = entering;
    end

    status = "MAX_ITER";
end

function [basisOut, ok] = removeArtificialFromBasis(Aaug, Aorig, basisIn, nOrig, tol)
    basis = basisIn(:);
    m = numel(basis);
    ok = true;

    for r = 1:m
        if basis(r) <= nOrig
            continue;
        end

        B = Aaug(:, basis);

        if rcond(B) < 1e-14
            ok = false;
            basisOut = basis;
            return;
        end

        tableauPart = B \ Aorig;
        currentOrigBasis = basis(basis <= nOrig);
        candidates = setdiff(1:nOrig, currentOrigBasis);

        pivotCol = [];
        for jj = candidates
            if abs(tableauPart(r, jj)) > tol
                pivotCol = jj;
                break;
            end
        end

        if isempty(pivotCol)
            ok = false;
            basisOut = basis;
            return;
        end

        basis(r) = pivotCol;
    end

    basisOut = basis;
end
