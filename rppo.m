% boxplot_rppo_tiempo.m
data_rl = [55.9, 45.3, 54.1, 55.2, 99.8, 61.1, 102.5, 46.8, 58.1, 62.8, 73.5, 57.6, 55.4, 57.0, 23.8];
bdot_ref = 189.0;

figure;
bp = boxplot(data_rl, 'Labels', {'RPPO'}, 'Widths', 0.5);
h = findobj(gca, 'Tag', 'Box');
patch(get(h, 'XData'), get(h, 'YData'), [0.29, 0.57, 0.89], 'FaceAlpha', 0.6);
hold on;
yline(bdot_ref, '--', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 1.5, ...
    'Label', sprintf('B-dot (%.2f)', bdot_ref), 'LabelHorizontalAlignment', 'left');
all_values = [data_rl(:); bdot_ref];
ylim([min(all_values) * 0.9, max(all_values) * 1.1]);
ylabel('Tiempo de primer cruce [min]');
title('Rapidez de detumbling');
grid on;
set(gca, 'FontSize', 11);
saveas(gcf, 'rapidez_detumbling_rppo.png');
% boxplot_rppo_mag.m
data_rl = [3.0235, 2.9403, 2.9908, 2.7419, 3.0914, 2.9086, 3.0369, 3.1339, 3.1394, 3.2299, 3.1603, 3.0683, 2.8838, 3.0135, 2.8931];
bdot_ref = 0.251;

figure;
bp = boxplot(data_rl, 'Labels', {'RPPO'}, 'Widths', 0.5);
h = findobj(gca, 'Tag', 'Box');
patch(get(h, 'XData'), get(h, 'YData'), [0.29, 0.57, 0.89], 'FaceAlpha', 0.6);
hold on;
yline(bdot_ref, '--', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 1.5, ...
    'Label', sprintf('B-dot (%.2f)', bdot_ref), 'LabelHorizontalAlignment', 'left');
all_values = [data_rl(:); bdot_ref];
ylim([min(all_values) * 0.9, max(all_values) * 1.1]);
ylabel('Uso medio de magnetopares [A·m²]');
title('Esfuerzo de actuación');
grid on;
set(gca, 'FontSize', 11);
saveas(gcf, 'esfuerzo_de_actuacion_rppo.png');
% boxplot_rppo_wfin.m
data_rl = [0.1877, 0.1907, 0.1799, 0.1732, 0.1920, 0.1956, 0.1653, 0.1692, 0.1913, 0.1974, 0.1879, 0.1828, 0.1864, 0.2136, 0.2593];
bdot_ref = 0.171;

figure;
bp = boxplot(data_rl, 'Labels', {'RPPO'}, 'Widths', 0.5);
h = findobj(gca, 'Tag', 'Box');
patch(get(h, 'XData'), get(h, 'YData'), [0.29, 0.57, 0.89], 'FaceAlpha', 0.6);
hold on;
yline(bdot_ref, '--', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 1.5, ...
    'Label', sprintf('B-dot (%.2f)', bdot_ref), 'LabelHorizontalAlignment', 'left');
all_values = [data_rl(:); bdot_ref];
ylim([min(all_values) * 0.9, max(all_values) * 1.1]);
ylabel('Velocidad angular final [°/s]');
title('Precisión asintótica');
grid on;
set(gca, 'FontSize', 11);
saveas(gcf, 'precision_asintotica_rppo.png');