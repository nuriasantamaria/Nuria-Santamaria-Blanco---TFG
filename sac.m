%% datos_historial_sac.m
% Datos SAC estudio historial: [w0, RL_wfin, Bdot_wfin, RL_mag, Bdot_mag, RL_t, Bdot_t]

SAC_HL0 = [
7.4329, 0.2320, 0.1500, 15.0807, 0.3241,  92.0, 177.7;
5.2439, 0.2862, 0.1439, 15.4725, 0.1947,  53.8, 115.0;
6.1478, 0.3537, 0.1985, 15.6546, 0.2714,  28.9, 226.8;
3.5572, 0.3203, 0.1627, 14.8449, 0.1738, 103.8, 190.7;
8.0432, 0.2673, 0.1955, 15.5026, 0.3591, 140.6, 213.6;
2.6270, 0.2061, 0.1820, 15.4426, 0.1439,  57.6, 190.9;
7.3344, 0.2394, 0.1477, 15.5965, 0.2760,  48.7, 169.8;
6.5154, 0.3527, 0.1982, 15.7033, 0.2766,  22.0, 167.8;
6.5631, 0.1524, 0.1521, 15.5818, 0.2736,  37.8, 156.4;
5.9183, 0.3447, 0.1799, 15.5207, 0.2863, 183.7, 277.8;
6.2334, 0.2578, 0.1707, 15.3109, 0.3030,  57.9, 277.2;
7.1480, 0.3476, 0.1793, 15.5190, 0.2526, 136.7, 105.8;
3.1879, 0.2836, 0.1644, 15.5572, 0.1727, 105.2, 226.6;
4.3662, 0.1649, 0.1484, 15.4142, 0.1928,  61.0, 168.7;
6.1965, 0.1781, 0.1973, 14.9011, 0.2602,  52.2, 170.9];

SAC_HL2 = [
7.4329, 0.1713, 0.1500,  8.2037, 0.3241, 186.9, 177.7;
5.2439, 0.1811, 0.1439,  7.7476, 0.1947, 148.3, 115.0;
6.1478, 0.2241, 0.1985,  7.6217, 0.2714,  15.4, 226.8;
3.5572, 0.1617, 0.1627,  7.9982, 0.1738, 193.4, 190.7;
8.0432, 0.2259, 0.1955,  7.7478, 0.3591, 102.2, 213.6;
2.6270, 0.1677, 0.1820,  7.7374, 0.1439, 192.7, 190.9;
7.3344, 0.1953, 0.1477,  7.6488, 0.2760,  18.9, 169.8;
6.5154, 0.2048, 0.1982,  7.7354, 0.2766,  13.2, 167.8;
6.5631, 0.2218, 0.1521,  7.7278, 0.2736,  27.4, 156.4;
5.9183, 0.1935, 0.1799,  7.6259, 0.2863,  13.4, 277.8;
6.2334, 0.2202, 0.1707,  7.8746, 0.3030, 151.3, 277.2;
7.1480, 0.1778, 0.1793,  7.8648, 0.2526, 192.4, 105.8;
3.1879, 0.2636, 0.1644,  7.6631, 0.1727, 148.0, 226.6;
4.3662, 0.1659, 0.1484,  7.6775, 0.1928, 147.4, 168.7;
6.1965, 0.2972, 0.1973,  7.8762, 0.2602, 149.7, 170.9];

SAC_HL4 = [
7.4329, 0.3709, 0.1500,  6.3394, 0.3241,  69.1, 177.7;
5.2439, 0.1871, 0.1439,  6.2529, 0.1947,  71.9, 115.0;
6.1478, 0.1804, 0.1985,  6.4792, 0.2714,  56.6, 226.8;
3.5572, 0.1747, 0.1627,  6.3579, 0.1738,  49.9, 190.7;
8.0432, 0.1928, 0.1955,  6.5660, 0.3591,  68.4, 213.6;
2.6270, 0.2694, 0.1820,  6.1589, 0.1439,  53.2, 190.9;
7.3344, 0.2089, 0.1477,  6.5422, 0.2760,  66.2, 169.8;
6.5154, 0.3713, 0.1982,  6.5363, 0.2766,  94.3, 167.8;
6.5631, 0.3710, 0.1521,  6.4422, 0.2736,  44.4, 156.4;
5.9183, 0.2449, 0.1799,  6.5745, 0.2863,  60.0, 277.8;
6.2334, 0.1774, 0.1707,  6.3869, 0.3030,  57.6, 277.2;
7.1480, 0.2196, 0.1793,  6.2802, 0.2526,  98.0, 105.8;
3.1879, 0.2070, 0.1644,  6.0942, 0.1727,  43.6, 226.6;
4.3662, 0.3708, 0.1484,  6.3805, 0.1928,  95.8, 168.7;
6.1965, 0.1729, 0.1973,  6.3818, 0.2602,  82.8, 170.9];

% n=8: 7 semillas no alcanzan el umbral; se asigna 680 min (límite horizonte)
SAC_HL8 = [
7.4329, 0.5298, 0.1500, 14.4174, 0.3241, 680.0, 177.7;
5.2439, 0.5447, 0.1439, 14.2253, 0.1947,  97.8, 115.0;
6.1478, 0.3815, 0.1985, 13.8888, 0.2714, 680.0, 226.8;
3.5572, 0.2230, 0.1627, 14.3196, 0.1738, 247.7, 190.7;
8.0432, 0.3472, 0.1955, 14.2500, 0.3591, 680.0, 213.6;
2.6270, 0.3121, 0.1820, 13.8430, 0.1439,  55.3, 190.9;
7.3344, 0.5091, 0.1477, 14.3764, 0.2760, 680.0, 169.8;
6.5154, 0.3808, 0.1982, 13.8834, 0.2766, 680.0, 167.8;
6.5631, 0.2138, 0.1521, 13.7845, 0.2736,  14.3, 156.4;
5.9183, 0.3799, 0.1799, 13.8432, 0.2863,  46.2, 277.8;
6.2334, 0.1953, 0.1707, 19.5714, 0.3030, 392.7, 277.2;
7.1480, 0.4232, 0.1793, 13.9163, 0.2526, 436.7, 105.8;
3.1879, 0.5382, 0.1644, 13.9608, 0.1727, 680.0, 226.6;
4.3662, 0.3700, 0.1484, 14.2939, 0.1928, 680.0, 168.7;
6.1965, 0.1849, 0.1973, 14.3623, 0.2602, 246.9, 170.9];

%% Datos mejor modelo SAC (sac_hl4, Optuna)
SAC_BEST = [
7.4329, 0.2071, 0.1500,  5.4378, 0.3241, 106.4, 177.7;
5.2439, 0.1821, 0.1439,  5.1079, 0.1947,  20.4, 115.0;
6.1478, 0.1840, 0.1985,  5.2201, 0.2714,  96.5, 226.8;
3.5572, 0.2157, 0.1627,  5.1120, 0.1738,  55.8, 190.7;
8.0432, 0.1789, 0.1955,  5.4796, 0.3591, 116.2, 213.6;
2.6270, 0.2262, 0.1820,  5.0980, 0.1439,  15.3, 190.9;
7.3344, 0.2046, 0.1477,  5.2816, 0.2760,  47.6, 169.8;
6.5154, 0.2071, 0.1982,  5.1793, 0.2766,  25.9, 167.8;
6.5631, 0.1941, 0.1521,  5.1946, 0.2736,  23.4, 156.4;
5.9183, 0.2069, 0.1799,  5.1900, 0.2863,  99.0, 277.8;
6.2334, 0.2043, 0.1707,  5.1898, 0.3030,  60.1, 277.2;
7.1480, 0.1791, 0.1793,  5.1676, 0.2526,  25.3, 105.8;
3.1879, 0.2120, 0.1644,  5.1067, 0.1727,  55.1, 226.6;
4.3662, 0.2026, 0.1484,  5.1981, 0.1928,  85.9, 168.7;
6.1965, 0.2283, 0.1973,  5.1393, 0.2602,  51.3, 170.9];


bdot_t   = mean(SAC_HL4(:,7));
bdot_mag = mean(SAC_HL4(:,5));
bdot_w   = mean(SAC_HL4(:,3));

c_gray = [0.70 0.70 0.70];
c_blue = [0.17 0.49 0.72];   
c_sac  = [0.16 0.64 0.36];   

labels_hl = {'k=0','k=2','k=4','k=8'};

function boxplot_hist(data_cell, labels, ylabel_str, title_str, bdot_ref, clrs)
    figure('Units','centimeters','Position',[5 5 14 9]);
    hold on; box on; grid on; grid minor;
    n = numel(data_cell);
    for k = 1:n
        x    = data_cell{k};
        q    = quantile(x,[0 0.25 0.5 0.75 1]);
        iqr_ = q(4)-q(2);
        lo   = max(q(1), q(2)-1.5*iqr_);
        hi   = min(q(5), q(4)+1.5*iqr_);
        fill([k-0.3 k+0.3 k+0.3 k-0.3 k-0.3],[q(2) q(2) q(4) q(4) q(2)], ...
             clrs{k},'FaceAlpha',0.75,'EdgeColor','k','LineWidth',0.8);
        plot([k-0.3 k+0.3],[q(3) q(3)],'k-','LineWidth',1.5);
        plot([k k],[lo q(2)],'k-','LineWidth',0.8);
        plot([k k],[q(4) hi],'k-','LineWidth',0.8);
        plot([k-0.15 k+0.15],[lo lo],'k-','LineWidth',0.8);
        plot([k-0.15 k+0.15],[hi hi],'k-','LineWidth',0.8);
        out = x(x<lo | x>hi);
        if ~isempty(out)
            plot(k*ones(size(out)),out,'ko','MarkerSize',4,'LineWidth',0.8);
        end
    end
    yline(bdot_ref,'--','Color',[0.85 0.33 0.10],'LineWidth',1.3,...
          'Label',sprintf('B-dot (%.2f)',bdot_ref),...
          'LabelHorizontalAlignment','left');
    xticks(1:n); xticklabels(labels);
    xlabel('Tamaño de historial'); ylabel(ylabel_str);
    title(title_str,'FontSize',11);
    xlim([0.5 n+0.5]); set(gca,'FontSize',10);
    hold off;
end

%% Función box plot individual (mejor modelo)
function boxplot_single(data, label, ylabel_str, title_str, bdot_ref, color)
    figure('Units','centimeters','Position',[5 5 8 9]);
    hold on; box on; grid on; grid minor;
    x = data;
    q = quantile(x,[0 0.25 0.5 0.75 1]);
    iqr_ = q(4)-q(2);
    lo = max(q(1), q(2)-1.5*iqr_);
    hi = min(q(5), q(4)+1.5*iqr_);
    fill([0.7 1.3 1.3 0.7 0.7],[q(2) q(2) q(4) q(4) q(2)],...
         color,'FaceAlpha',0.75,'EdgeColor','k','LineWidth',0.8);
    plot([0.7 1.3],[q(3) q(3)],'k-','LineWidth',1.5);
    plot([1 1],[lo q(2)],'k-','LineWidth',0.8);
    plot([1 1],[q(4) hi],'k-','LineWidth',0.8);
    plot([0.85 1.15],[lo lo],'k-','LineWidth',0.8);
    plot([0.85 1.15],[hi hi],'k-','LineWidth',0.8);
    out = x(x<lo | x>hi);
    if ~isempty(out)
        plot(ones(size(out)),out,'ko','MarkerSize',4,'LineWidth',0.8);
    end
    yline(bdot_ref,'--','Color',[0.85 0.33 0.10],'LineWidth',1.3,...
          'Label',sprintf('B-dot (%.2f)',bdot_ref),...
          'LabelHorizontalAlignment','left');
    xticks(1); xticklabels({label});
    ylabel(ylabel_str); title(title_str,'FontSize',11);
    xlim([0.5 1.5]); set(gca,'FontSize',10);
    hold off;
end


%% PARTE 1 — Estudio de historial SAC (n=4 en azul)

clrs_hl = {c_gray, c_gray, c_blue, c_gray};

% Figura 1 — Rapidez
boxplot_hist({SAC_HL0(:,6), SAC_HL2(:,6), SAC_HL4(:,6), SAC_HL8(:,6)}, ...
    labels_hl, 'Tiempo de primer cruce [min]', ...
    'Rapidez de detumbling', bdot_t, clrs_hl);
exportgraphics(gcf,'sac_box_historial_tiempo.pdf','ContentType','vector');

% Figura 2 — Magnetopares
boxplot_hist({SAC_HL0(:,4), SAC_HL2(:,4), SAC_HL4(:,4), SAC_HL8(:,4)}, ...
    labels_hl, 'Uso medio de magnetopares [A·m²]', ...
    'Esfuerzo de actuación', bdot_mag, clrs_hl);
exportgraphics(gcf,'sac_box_historial_mag.pdf','ContentType','vector');

% Figura 3 — Velocidad angular final
boxplot_hist({SAC_HL0(:,2), SAC_HL2(:,2), SAC_HL4(:,2), SAC_HL8(:,2)}, ...
    labels_hl, 'Velocidad angular final [°/s]', ...
    'Precisión asintótica', bdot_w, clrs_hl);
exportgraphics(gcf,'sac_box_historial_wfin.pdf','ContentType','vector');


%% PARTE 2 — Mejor modelo SAC (sac_hl4 Optuna, en verde)

bdot_t_best   = mean(SAC_BEST(:,7));
bdot_mag_best = mean(SAC_BEST(:,5));
bdot_w_best   = mean(SAC_BEST(:,3));

%% Figuras SAC best model
boxplot_single(SAC_BEST(:,6), 'SAC', 'Tiempo de primer cruce [min]', ...
    'Rapidez de detumbling — SAC', bdot_t_best, c_sac);
exportgraphics(gcf,'sac_box_best_tiempo.pdf','ContentType','vector');


boxplot_single(SAC_BEST(:,4), 'SAC', 'Uso medio de magnetopares [A·m²]', ...
    'Esfuerzo de actuación — SAC', bdot_mag_best, c_sac);
exportgraphics(gcf,'sac_box_best_mag.pdf','ContentType','vector');

boxplot_single(SAC_BEST(:,2), 'SAC', 'Velocidad angular final [°/s]', ...
    'Precisión asintótica — SAC', bdot_w_best, c_sac);
exportgraphics(gcf,'sac_box_best_wfin.pdf','ContentType','vector');