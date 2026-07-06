%% datos_historial.m
% Datos: cada fila es una semilla, columnas: [w0, RL_wfin, Bdot_wfin, RL_mag, Bdot_mag, RL_t, Bdot_t]

HL0 = [
7.4329,0.1937,0.1500,4.4852,0.3241,104.8,177.7;
5.2439,0.1911,0.1439,4.3734,0.1947, 28.4,115.0;
6.1478,0.2009,0.1985,4.2607,0.2714, 62.5,226.8;
3.5572,0.3186,0.1627,5.2876,0.1738, 48.4,190.7;
8.0432,0.1917,0.1955,4.4632,0.3591, 62.4,213.6;
2.6270,0.1974,0.1820,4.1236,0.1439, 58.0,190.9;
7.3344,0.2067,0.1477,4.4108,0.2760, 22.2,169.8;
6.5154,0.2203,0.1982,4.4055,0.2766, 39.2,167.8;
6.5631,0.2343,0.1521,4.3696,0.2736, 52.7,156.4;
5.9183,0.1765,0.1799,4.3982,0.2863, 56.9,277.8;
6.2334,0.2084,0.1707,4.3655,0.3030, 54.0,277.2;
7.1480,0.2640,0.1793,4.4310,0.2526, 67.4,105.8;
3.1879,0.2091,0.1644,4.2637,0.1727, 13.2,226.6;
4.3662,0.1766,0.1484,4.3020,0.1928,105.3,168.7;
6.1965,0.2027,0.1973,4.3842,0.2602, 54.0,170.9];

HL2 = [
7.4329,0.1964,0.1500,1.7838,0.3241, 61.7,177.7;
5.2439,0.1823,0.1439,1.5075,0.1947, 34.9,115.0;
6.1478,0.1664,0.1985,1.5933,0.2714, 59.0,226.8;
3.5572,0.2196,0.1627,1.5126,0.1738, 49.6,190.7;
8.0432,0.1827,0.1955,1.8250,0.3591, 63.8,213.6;
2.6270,0.1606,0.1820,1.3720,0.1439, 19.7,190.9;
7.3344,0.1945,0.1477,1.6977,0.2760, 99.5,169.8;
6.5154,0.1587,0.1982,1.4754,0.2766, 17.9,167.8;
6.5631,0.2060,0.1521,1.5447,0.2736, 23.2,156.4;
5.9183,0.1602,0.1799,1.6573,0.2863, 58.5,277.8;
6.2334,0.2061,0.1707,1.5880,0.3030, 57.6,277.2;
7.1480,0.1585,0.1793,1.5100,0.2526, 24.7,105.8;
3.1879,0.1598,0.1644,1.3449,0.1727, 53.6,226.6;
4.3662,0.1729,0.1484,1.6046,0.1928, 56.0,168.7;
6.1965,0.1745,0.1973,1.6213,0.2602, 56.0,170.9];

HL4 = [
7.4329,0.1988,0.1500,6.5170,0.3241, 97.7,177.7;
5.2439,0.2229,0.1439,6.4576,0.1947,107.2,115.0;
6.1478,0.1812,0.1985,6.5571,0.2714, 44.8,226.8;
3.5572,0.1749,0.1627,6.5480,0.1738, 54.7,190.7;
8.0432,0.1907,0.1955,6.6464,0.3591, 52.1,213.6;
2.6270,0.2393,0.1820,6.2776,0.1439,143.6,190.9;
7.3344,0.2402,0.1477,6.4407,0.2760,149.6,169.8;
6.5154,0.1840,0.1982,6.6522,0.2766, 42.8,167.8;
6.5631,0.1832,0.1521,6.6532,0.2736, 30.0,156.4;
5.9183,0.1878,0.1799,6.6594,0.2863, 54.0,277.8;
6.2334,0.2177,0.1707,6.6008,0.3030, 45.2,277.2;
7.1480,0.2303,0.1793,6.5323,0.2526, 45.0,105.8;
3.1879,0.2376,0.1644,5.7611,0.1727, 16.8,226.6;
4.3662,0.1756,0.1484,6.6008,0.1928, 58.0,168.7;
6.1965,0.1674,0.1973,6.6792,0.2602, 47.4,170.9];

HL8 = [
7.4329,0.1796,0.1500,6.9869,0.3241,102.9,177.7;
5.2439,0.1548,0.1439,8.4859,0.1947, 22.1,115.0;
6.1478,0.1929,0.1985,6.6684,0.2714, 54.5,226.8;
3.5572,0.1814,0.1627,7.5385,0.1738, 60.2,190.7;
8.0432,0.2099,0.1955,9.1290,0.3591, 66.7,213.6;
2.6270,0.1923,0.1820,8.0585,0.1439, 18.9,190.9;
7.3344,0.1987,0.1477,8.9675,0.2760, 45.2,169.8;
6.5154,0.1695,0.1982,6.6472,0.2766, 51.6,167.8;
6.5631,0.1643,0.1521,8.4564,0.2736, 55.2,156.4;
5.9183,0.1468,0.1799,6.7593,0.2863, 55.0,277.8;
6.2334,0.1521,0.1707,6.8421,0.3030,107.2,277.2;
7.1480,0.2684,0.1793,6.5231,0.2526, 47.2,105.8;
3.1879,0.1727,0.1644,7.1314,0.1727, 22.5,226.6;
4.3662,0.1958,0.1484,6.7952,0.1928, 66.9,168.7;
6.1965,0.1586,0.1973,7.0801,0.2602, 29.4,170.9];


bdot_t   = mean(HL0(:,7));   
bdot_mag = mean(HL0(:,5));   
bdot_w   = mean(HL0(:,3));   


c_gray = [0.70 0.70 0.70];
c_blue = [0.17 0.49 0.72];  
clrs   = {c_gray, c_blue, c_gray, c_gray};

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
    
        fill([k-0.3 k+0.3 k+0.3 k-0.3 k-0.3], [q(2) q(2) q(4) q(4) q(2)], ...
             clrs{k}, 'FaceAlpha',0.75, 'EdgeColor','k','LineWidth',0.8);
      
        plot([k-0.3 k+0.3],[q(3) q(3)],'k-','LineWidth',1.5);
       
        plot([k k],[lo q(2)],'k-','LineWidth',0.8);
        plot([k k],[q(4) hi],'k-','LineWidth',0.8);
        plot([k-0.15 k+0.15],[lo lo],'k-','LineWidth',0.8);
        plot([k-0.15 k+0.15],[hi hi],'k-','LineWidth',0.8);
        % outliers
        out = x(x<lo | x>hi);
        if ~isempty(out)
            plot(k*ones(size(out)), out, 'ko','MarkerSize',4,'LineWidth',0.8);
        end
    end
 
    yline(bdot_ref,'--','Color',[0.85 0.33 0.10],'LineWidth',1.3, ...
          'Label',sprintf('B-dot (%.2f)',bdot_ref),'LabelHorizontalAlignment','left');
    xticks(1:n); xticklabels(labels);
    xlabel('Tamaño de historial'); ylabel(ylabel_str);
    title(title_str,'FontSize',11);
    xlim([0.5 n+0.5]); set(gca,'FontSize',10);
    hold off;
end

labels = {'k=0','k=2','k=4','k=8'};

%% Figura 1 — Tiempo de cruce
boxplot_hist({HL0(:,6), HL2(:,6), HL4(:,6), HL8(:,6)}, labels, ...
    'Tiempo de primer cruce [min]', 'Rapidez de detumbling', bdot_t, clrs);
exportgraphics(gcf,'box_tiempo.pdf','ContentType','vector');

%% Figura 2 — Magnetopares
boxplot_hist({HL0(:,4), HL2(:,4), HL4(:,4), HL8(:,4)}, labels, ...
    'Uso medio de magnetopares [A·m²]', 'Esfuerzo de actuación', bdot_mag, clrs);
exportgraphics(gcf,'box_mag.pdf','ContentType','vector');

%% Figura 3 — Velocidad angular final
boxplot_hist({HL0(:,2), HL2(:,2), HL4(:,2), HL8(:,2)}, labels, ...
    'Velocidad angular final [°/s]', 'Precisión asintótica', bdot_w, clrs);
exportgraphics(gcf,'box_wfin.pdf','ContentType','vector');

%% datos_ppo_best.m
% PPO n=2 best_model: [w0, RL_wfin, Bdot_wfin, RL_mag, Bdot_mag, RL_t, Bdot_t]

PPO_BEST = [
7.4329, 0.1964, 0.1500,  1.7838, 0.3241,  61.7, 177.7;
5.2439, 0.1823, 0.1439,  1.5075, 0.1947,  34.9, 115.0;
6.1478, 0.1664, 0.1985,  1.5933, 0.2714,  59.0, 226.8;
3.5572, 0.2196, 0.1627,  1.5126, 0.1738,  49.6, 190.7;
8.0432, 0.1827, 0.1955,  1.8250, 0.3591,  63.8, 213.6;
2.6270, 0.1605, 0.1820,  1.3720, 0.1439,  19.7, 190.9;
7.3344, 0.1945, 0.1477,  1.6977, 0.2760,  99.5, 169.8;
6.5154, 0.1587, 0.1982,  1.4754, 0.2766,  17.9, 167.8;
6.5631, 0.2060, 0.1521,  1.5447, 0.2736,  23.2, 156.4;
5.9183, 0.1602, 0.1799,  1.6573, 0.2863,  58.5, 277.8;
6.2334, 0.2061, 0.1707,  1.5880, 0.3030,  57.6, 277.2;
7.1480, 0.1585, 0.1793,  1.5100, 0.2526,  24.7, 105.8;
3.1879, 0.1598, 0.1644,  1.3449, 0.1727,  53.6, 226.6;
4.3662, 0.1729, 0.1484,  1.6046, 0.1928,  56.0, 168.7;
6.1965, 0.1745, 0.1973,  1.6213, 0.2602,  56.0, 170.9];


bdot_t   = mean(PPO_BEST(:,7));
bdot_mag = mean(PPO_BEST(:,5));
bdot_w   = mean(PPO_BEST(:,3));


c_ppo = [0.85 0.33 0.10];  % naranja

%% Función box plot individual (reutiliza la de rppo/sac)
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

%% Figuras PPO best model
boxplot_single(PPO_BEST(:,6), 'PPO', 'Tiempo de primer cruce [min]', ...
    'Rapidez de detumbling — PPO (n=2)', bdot_t, c_ppo);
exportgraphics(gcf,'ppo_box_best_tiempo.pdf','ContentType','vector');

boxplot_single(PPO_BEST(:,4), 'PPO', 'Uso medio de magnetopares [A·m²]', ...
    'Esfuerzo de actuación — PPO (n=2)', bdot_mag, c_ppo);
exportgraphics(gcf,'ppo_box_best_mag.pdf','ContentType','vector');

boxplot_single(PPO_BEST(:,2), 'PPO', 'Velocidad angular final [°/s]', ...
    'Precisión asintótica — PPO (n=2)', bdot_w, c_ppo);
exportgraphics(gcf,'ppo_box_best_wfin.pdf','ContentType','vector');